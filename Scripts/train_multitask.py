"""Entrenamiento PPO sobre DUMMultitaskEnv (v14: head + wave + grip).

Hace warm-start opcional desde la policy v13 (final.zip) copiando solo los pesos
con shape compatible. Esto reusa el "saber mirar" ya entrenado en v13.

Uso:
    # Smoke (50k steps, 8 envs, ~5 min): valida pipeline.
    python Scripts/train_multitask.py --steps 50000 --n-envs 8 --name v14_smoke

    # Training largo (6M steps, 16 envs) — lo lanza el usuario cuando decida:
    python Scripts/train_multitask.py --steps 6000000 --n-envs 16 --n-steps 512 \
        --batch-size 256 --n-epochs 10 --net-arch "128,128" --ent-coef 0.005 \
        --learning-rate 3e-4 --name ppo_dum_v14
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Limitar threads de PyTorch para no pelear con los workers de MuJoCo.
import torch
torch.set_num_threads(2)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from rl_env_multitask import DUMMultitaskEnv

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
V13_FINAL = RUNS_DIR / "ppo_dum_v13" / "final.zip"


def make_env(seed=0, log_dir=None):
    def _init():
        env = DUMMultitaskEnv()
        env.reset(seed=seed)
        if log_dir is not None:
            env = Monitor(env, str(log_dir / f"monitor_{seed}.csv"))
        return env

    return _init


def linear_schedule(initial: float, final: float):
    """SB3 le pasa al schedule un escalar 'progress_remaining' que va de 1.0 a 0.0.

    Para que el LR baje linealmente de initial a final hay que devolver:
        final + (initial - final) * progress
    """

    def _f(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return _f


def warm_start_from_v13(model: PPO, v13_path: Path) -> dict:
    """Copia los pesos compatibles desde la policy v13 a la policy v14.

    Devuelve un dict con stats: keys totales, copiadas, skipped (con motivo).
    """
    stats = {"total": 0, "copied": 0, "skipped_missing": [], "skipped_shape": []}
    if not v13_path.exists():
        print(f"[warm-start] No se encontro {v13_path} — entrenando desde cero.")
        stats["skipped_missing"].append(str(v13_path))
        return stats

    print(f"[warm-start] Cargando v13 desde {v13_path}")
    old = PPO.load(str(v13_path), device="cpu")
    new_sd = model.policy.state_dict()
    old_sd = old.policy.state_dict()

    merged = {}
    for k, v_new in new_sd.items():
        stats["total"] += 1
        if k in old_sd:
            v_old = old_sd[k]
            if v_old.shape == v_new.shape:
                merged[k] = v_old
                stats["copied"] += 1
            else:
                merged[k] = v_new
                stats["skipped_shape"].append(
                    f"{k}: new={tuple(v_new.shape)} vs old={tuple(v_old.shape)}"
                )
        else:
            merged[k] = v_new
            stats["skipped_missing"].append(k)

    model.policy.load_state_dict(merged, strict=False)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--smoke", action="store_true",
                   help="Override: 50k steps, 4 envs.")
    p.add_argument("--resume", type=str, default=None,
                   help="Reanudar desde un checkpoint .zip (no hace warm-start desde v13).")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="LR inicial del schedule lineal (final = LR/3).")
    p.add_argument("--learning-rate-final", type=float, default=1e-4,
                   help="LR final del schedule lineal.")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--net-arch", type=str, default="128,128",
                   help="MLP de pi y vf. v14 usa 128,128 por defecto para preservar "
                        "el warm-start de v13.")
    p.add_argument("--no-warm-start", action="store_true",
                   help="Entrenar desde cero (saltea la copia desde v13).")
    args = p.parse_args()

    if args.smoke:
        args.steps = 50_000
        args.n_envs = 4

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"ppo_dum_multitask_{timestamp}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    if args.n_envs == 1:
        env = DummyVecEnv([make_env(seed=0, log_dir=run_dir)])
    else:
        env = SubprocVecEnv(
            [make_env(seed=i, log_dir=run_dir) for i in range(args.n_envs)]
        )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
        from_scratch = False
        warm_stats = None
    else:
        net_arch = [int(x) for x in args.net_arch.split(",") if x.strip()]
        lr_sched = linear_schedule(args.learning_rate, args.learning_rate_final)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            learning_rate=lr_sched,
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

        if args.no_warm_start:
            print("[warm-start] Saltado por --no-warm-start.")
            warm_stats = None
        else:
            warm_stats = warm_start_from_v13(model, V13_FINAL)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.steps // 10 // args.n_envs, 1),
        save_path=str(run_dir),
        name_prefix="checkpoint",
    )

    started_at = datetime.now()
    print("=" * 70)
    print(f"INICIO:        {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run:           {name}")
    print(f"Modo:          {'FROM SCRATCH (red nueva)' if from_scratch else 'RESUMED'}")
    print(f"Steps:         {args.steps:,}")
    print(f"N envs:        {args.n_envs}")
    print(f"n_steps/env:   {args.n_steps}  ->  rollout total = {args.n_steps * args.n_envs}")
    print(f"batch_size:    {args.batch_size}")
    print(f"n_epochs:      {args.n_epochs}")
    print(f"net_arch:      {args.net_arch}")
    print(f"learning_rate: {args.learning_rate} -> {args.learning_rate_final} (linear)")
    print(f"ent_coef:      {args.ent_coef}")
    print(f"torch threads: {torch.get_num_threads()}")
    if warm_stats is not None:
        print(
            f"warm-start v13: copied={warm_stats['copied']}/{warm_stats['total']}  "
            f"skipped_shape={len(warm_stats['skipped_shape'])}  "
            f"skipped_missing={len(warm_stats['skipped_missing'])}"
        )
        for s in warm_stats["skipped_shape"]:
            print(f"   shape mismatch: {s}")
    print("=" * 70, flush=True)

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_cb,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - t0

    final_path = run_dir / "final.zip"
    model.save(str(final_path))
    finished_at = datetime.now()

    print("=" * 70)
    print(f"INICIO:        {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"FIN:           {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DURACION:      {timedelta(seconds=int(elapsed))} ({elapsed:.1f} s)")
    print(f"Modelo final:  {final_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
