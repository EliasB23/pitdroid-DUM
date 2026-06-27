"""Entrenamiento PPO sobre DUMHeadTrackingEnv usando Stable-Baselines3.

Uso:
    # Smoke training: 20k steps single-env para validar pipeline
    python train_ppo.py --smoke
    # Training real: 2M steps con 8 envs paralelos
    python train_ppo.py --steps 2000000 --n-envs 8
    # Resumir desde checkpoint
    python train_ppo.py --steps 2000000 --resume runs/ppo_dum_xxx/checkpoint.zip
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Limitar threads de PyTorch para no pelear con los workers MuJoCo
# (recomendado por el experto: 16 logical CPUs / 8 fisicos -> torch en 2, libra 6 para envs)
import torch
torch.set_num_threads(2)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_env import DUMHeadTrackingEnv

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


class HeadCurriculumV14Callback(BaseCallback):
    """v14_head: doble curriculum durante el resume desde v13.

    1. Cone ramp (0 → cone_ramp_steps): ampliacion lineal del cono de target
       desde (75°,45°) hasta (150°,50°). v13 entreno con (75°,45°) y nunca vio
       targets mas alla — entonces necesita exposicion gradual al area extendida.
    2. Ent_coef step (a ent_switch_steps): empieza alto (0.03) para forzar
       exploracion en dimensiones del action space (esp. HeadRot) que v13 dejo
       quietas, baja a 0.005 cuando ya tuvo tiempo de salir del local optimum.
    """
    def __init__(self, total_timesteps: int,
                 cone_start_az_deg: float = 75.0, cone_end_az_deg: float = 150.0,
                 cone_start_el_deg: float = 45.0, cone_end_el_deg: float = 50.0,
                 cone_ramp_steps: int = 7_000_000,
                 ent_high: float = 0.03, ent_low: float = 0.005,
                 ent_switch_steps: int = 5_000_000,
                 log_every: int = 1_000_000, verbose: int = 0):
        super().__init__(verbose)
        self.total = int(total_timesteps)
        self.cone_start_az = float(np.deg2rad(cone_start_az_deg))
        self.cone_end_az = float(np.deg2rad(cone_end_az_deg))
        self.cone_start_el = float(np.deg2rad(cone_start_el_deg))
        self.cone_end_el = float(np.deg2rad(cone_end_el_deg))
        self.cone_ramp_steps = int(cone_ramp_steps)
        self.ent_high = float(ent_high)
        self.ent_low = float(ent_low)
        self.ent_switch_steps = int(ent_switch_steps)
        self.log_every = int(log_every)
        self._last_logged = 0
        self._last_ent = None
        self._switched_ent = False

    def _on_step(self) -> bool:
        # Cone ramp
        if self.num_timesteps < self.cone_ramp_steps:
            t = self.num_timesteps / self.cone_ramp_steps
            cone_az = self.cone_start_az + t * (self.cone_end_az - self.cone_start_az)
            cone_el = self.cone_start_el + t * (self.cone_end_el - self.cone_start_el)
        else:
            cone_az = self.cone_end_az
            cone_el = self.cone_end_el
        try:
            self.training_env.env_method("set_cone_limits", cone_az, cone_el)
        except Exception:
            pass

        # ent_coef step transition
        if self.num_timesteps < self.ent_switch_steps:
            target_ent = self.ent_high
        else:
            target_ent = self.ent_low
        if self._last_ent != target_ent:
            self.model.ent_coef = target_ent
            self._last_ent = target_ent
            if self._switched_ent or self.num_timesteps > 1000:
                from datetime import datetime
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"[v14_head {stamp}] ent_coef set to {target_ent} @ steps={self.num_timesteps:,}",
                      flush=True)
            self._switched_ent = True

        # Log periodico
        if self.verbose and (self.num_timesteps - self._last_logged) >= self.log_every:
            from datetime import datetime
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[v14_head {stamp}] steps={self.num_timesteps:,} "
                  f"cone={np.rad2deg(cone_az):.0f}°az/{np.rad2deg(cone_el):.0f}°el "
                  f"ent_coef={self.model.ent_coef:.4f}",
                  flush=True)
            self._last_logged = self.num_timesteps
        return True


def make_env(seed=0, log_dir=None, cone_az_deg=None, cone_el_deg=None,
             w_headrot_guidance=0.0, w_headbase_tilt_penalty=0.0):
    def _init():
        rw = {}
        if w_headrot_guidance > 0.0:
            rw["w_headrot_guidance"] = w_headrot_guidance
        if w_headbase_tilt_penalty > 0.0:
            rw["w_headbase_tilt_penalty"] = w_headbase_tilt_penalty
        env = DUMHeadTrackingEnv(
            reward_weights=rw if rw else None,
            cone_az_deg=cone_az_deg,
            cone_el_deg=cone_el_deg,
        )
        env.reset(seed=seed)
        if log_dir is not None:
            env = Monitor(env, str(log_dir / f"monitor_{seed}.csv"))
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--smoke", action="store_true",
                   help="Override: 20k steps, 1 env, para validar pipeline")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="Coeficiente de entropia en la loss de PPO. 0 = sin exploracion forzada. "
                        "Subir a 0.005-0.01 si la policy converge prematuramente a local optimum.")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Rollout por env antes de cada update. Total = n_steps * n_envs.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Minibatch size para SGD. Subir con mas envs (256 si n_envs>=8).")
    p.add_argument("--n-epochs", type=int, default=10,
                   help="Numero de passes sobre el rollout por update.")
    p.add_argument("--net-arch", type=str, default="64,64",
                   help="Arquitectura del MLP (pi y vf). Ejemplo: '128,128' o '256,256,128'.")
    p.add_argument("--head-v14-curriculum", action="store_true",
                   help="Activa curriculum v14_head: cone ramp 75°→150° az + ent_coef step 0.03→0.005")
    p.add_argument("--v14-cone-ramp-steps", type=int, default=7_000_000)
    p.add_argument("--v14-ent-switch-steps", type=int, default=5_000_000)
    p.add_argument("--v14-cone-end-az", type=float, default=150.0,
                   help="Cono azimuth final en grados (default 150°, max fisico 160°)")
    p.add_argument("--v14-cone-end-el", type=float, default=50.0,
                   help="Cono elevation final en grados (default 50°)")
    # v14b: training from scratch con cono fijo + reward shaping
    p.add_argument("--cone-az-deg", type=float, default=None,
                   help="(v14b) Cono azimuth en grados, fijo desde el inicio")
    p.add_argument("--cone-el-deg", type=float, default=None,
                   help="(v14b) Cono elevation en grados, fijo desde el inicio")
    p.add_argument("--w-headrot-guidance", type=float, default=0.0,
                   help="(v14b) Penalty weight: cuando target_az > 17°, penaliza HeadRot_qpos lejos del target_az. v14c usa 2.0.")
    p.add_argument("--w-headbase-tilt-penalty", type=float, default=0.0,
                   help="(v14c) Penalty directo a |HeadBase_qpos| cuando target_az > 30°. Fuerza a NO usar HeadBase para compensar yaw. v14c usa 1.5.")
    args = p.parse_args()

    if args.smoke:
        args.steps = 20_000
        args.n_envs = 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"ppo_dum_{timestamp}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    # Crear envs paralelos
    env_kwargs = dict(
        cone_az_deg=args.cone_az_deg,
        cone_el_deg=args.cone_el_deg,
        w_headrot_guidance=args.w_headrot_guidance,
        w_headbase_tilt_penalty=args.w_headbase_tilt_penalty,
    )
    if args.n_envs == 1:
        env = DummyVecEnv([make_env(seed=0, log_dir=run_dir, **env_kwargs)])
    else:
        env = SubprocVecEnv([make_env(seed=i, log_dir=run_dir, **env_kwargs) for i in range(args.n_envs)])

    # Hiperparams PPO tipo MuJoCo continuous control
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
        from_scratch = False
    else:
        # Parse net-arch "a,b,c" -> [a,b,c]
        net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            policy_kwargs=dict(net_arch=dict(pi=net_arch, vf=net_arch)),
        )
        from_scratch = True

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.steps // 10 // args.n_envs, 1),
        save_path=str(run_dir),
        name_prefix="checkpoint",
    )
    callbacks = [checkpoint_cb]
    if args.head_v14_curriculum:
        callbacks.append(HeadCurriculumV14Callback(
            total_timesteps=args.steps,
            cone_end_az_deg=args.v14_cone_end_az,
            cone_end_el_deg=args.v14_cone_end_el,
            cone_ramp_steps=args.v14_cone_ramp_steps,
            ent_switch_steps=args.v14_ent_switch_steps,
            verbose=1,
        ))

    # === Banner de inicio ===
    started_at = datetime.now()
    print("=" * 70)
    print(f"INICIO:        {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run:           {name}")
    print(f"Modo:          {'FROM SCRATCH (red nueva, pesos random)' if from_scratch else 'RESUMED'}")
    print(f"Steps:         {args.steps:,}")
    print(f"N envs:        {args.n_envs}")
    print(f"n_steps/env:   {args.n_steps}  ->  rollout total = {args.n_steps * args.n_envs}")
    print(f"batch_size:    {args.batch_size}")
    print(f"n_epochs:      {args.n_epochs}")
    print(f"net_arch:      {args.net_arch}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"ent_coef:      {args.ent_coef}")
    print(f"torch threads: {torch.get_num_threads()}")
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - t0

    final_path = run_dir / "final.zip"
    model.save(str(final_path))
    finished_at = datetime.now()

    # === Banner de fin ===
    print("=" * 70)
    print(f"INICIO:        {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"FIN:           {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DURACION:      {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)")
    print(f"Modelo final:  {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
