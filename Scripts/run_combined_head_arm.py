"""Runtime combinado: v13 (head tracking) + v7e (arm grab) en una sola sim.

La cabeza apunta a la BOLA AMARILLA (en lugar del target rojo mocap). El brazo
trata de agarrarla. Ambas policies leen sus obs de la MISMA MjData compartida.

Genera un video MP4 mostrando el comportamiento con varias alturas de spawn
para ver cuanto generaliza v7e fuera de su distribucion de training (offset
maximo en training fue 17cm, aca lo subimos progresivamente).

Uso:
    python Scripts/run_combined_head_arm.py [--episodes 6] [--video out.mp4]
"""
import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import DUMHeadTrackingEnv
from rl.envs.grab_env import DUMGrabEnv


class CombinedHeadArmEnv(DUMGrabEnv):
    """DUMGrabEnv extendido: en cada step, ANTES de aplicar el ctrl del brazo,
    computa la obs del head (con target = yellow_ball), pasa por VecNormalize de v13,
    predice la accion del head y mete su ctrl en los actuadores correspondientes.

    El target_world del head policy es la posicion de la bola amarilla.
    """

    def __init__(self, head_model, head_vn, **kwargs):
        super().__init__(**kwargs)
        self.head_model = head_model
        self.head_vn = head_vn

        # IDs/addrs para computar obs del head desde la MjData del grab env
        self._head_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'LenteExt_link')
        self._lens_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'lens_center')
        head_jnames = ['Neck_joint', 'HeadBase_joint', 'HeadRotation_joint']
        head_jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in head_jnames]
        self._head_qpos_adr_c = np.array([self.model.jnt_qposadr[j] for j in head_jids])
        self._head_dof_adr_c = np.array([self.model.jnt_dofadr[j] for j in head_jids])
        hb_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'HipBody_joint')
        self._hb_qpos_adr_c = self.model.jnt_qposadr[hb_jid]
        self._hb_dof_adr_c = self.model.jnt_dofadr[hb_jid]
        head_anames = ['act_Neck', 'act_HeadBase', 'act_HeadRot']
        self._head_act_ids_c = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                                for n in head_anames]
        self._head_ctrl_lo_c = np.array([self.model.actuator_ctrlrange[a, 0] for a in self._head_act_ids_c])
        self._head_ctrl_hi_c = np.array([self.model.actuator_ctrlrange[a, 1] for a in self._head_act_ids_c])

        # Estado para tracking de velocidad del target (bola)
        self._target_world_prev = None
        self._prev_head_action = np.zeros(3, dtype=np.float32)
        # Datos del HEAD policy para la obs (los necesita pero no los chequea)
        self._head_theta_log = 0.0  # para visualizar

    def _compute_head_obs(self):
        """Replica DUMHeadTrackingEnv._get_obs operando sobre nuestra MjData."""
        ball_pos = self._ball_pos_world()
        target_world = ball_pos
        if self._target_world_prev is None:
            self._target_world_prev = target_world.copy()
        dt_ctrl = self.frame_skip * float(self.model.opt.timestep)
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        lens_origin = self.data.site_xpos[self._lens_site_id]
        tloc = head_mat.T @ (target_world - lens_origin)
        tdist = float(np.linalg.norm(tloc))
        tdir = tloc / (tdist + 1e-9)
        tvel_world = (target_world - self._target_world_prev) / dt_ctrl
        tvel = head_mat.T @ tvel_world

        qpos_head = self.data.qpos[self._head_qpos_adr_c].copy()
        qvel_head = self.data.qvel[self._head_dof_adr_c].copy()
        qpos_hb = float(self.data.qpos[self._hb_qpos_adr_c])
        qvel_hb = float(self.data.qvel[self._hb_dof_adr_c])

        obs = np.concatenate([
            qpos_head, qvel_head,           # 6
            [qpos_hb, qvel_hb],              # 2
            tdir, [tdist],                   # 4
            tvel,                             # 3
            self._prev_head_action.astype(np.float64),  # 3
        ])
        # Para overlay del video — angulo entre head forward y direccion a target
        # HEAD_FORWARD_LOCAL = (0, -1, 0) en frame local del LenteExt_link
        head_forward_local = np.array([0.0, -1.0, 0.0])
        forward_world = head_mat @ head_forward_local
        if tdist > 1e-6:
            delta = (target_world - lens_origin) / tdist
            cos_a = float(np.clip(np.dot(forward_world, delta), -1.0, 1.0))
            self._head_theta_log = float(np.arccos(cos_a))
        return obs

    def _apply_action(self, arm_action):
        # Ctrl del brazo (parent class)
        full_ctrl = super()._apply_action(arm_action)
        # Obs del head -> (opcionalmente normalizada) -> action
        head_obs = self._compute_head_obs()
        if self.head_vn is not None:
            head_obs_input = self.head_vn.normalize_obs(head_obs[None, :])
        else:
            head_obs_input = head_obs[None, :]
        head_action, _ = self.head_model.predict(head_obs_input, deterministic=True)
        head_action = head_action[0]
        # Mapear [-1,1] a ctrlrange
        a = np.clip(head_action, -1.0, 1.0).astype(np.float64)
        head_ctrl = (a + 1.0) * 0.5 * (self._head_ctrl_hi_c - self._head_ctrl_lo_c) + self._head_ctrl_lo_c
        for i, aid in enumerate(self._head_act_ids_c):
            full_ctrl[aid] = head_ctrl[i]
        # Guardar prev_action y prev_target
        self._prev_head_action = head_action.astype(np.float32).copy()
        self._target_world_prev = self._ball_pos_world().copy()
        return full_ctrl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--video", type=str, default="runs/grab_phase1_v7e/combined_head_arm.mp4")
    p.add_argument("--head-policy", type=str, default="runs/ppo_dum_v13/final.zip")
    p.add_argument("--arm-policy", type=str, default="runs/grab_phase1_v7e/final.zip")
    p.add_argument("--enable-throw", action="store_true",
                   help="Activar HELD->THROWN post-grab (v9)")
    args = p.parse_args()

    # Cargar HEAD policy (v13 fue entrenado sin VecNormalize, asi que pasamos obs cruda)
    print("[setup] Cargando head policy v13...")
    head_vn_path = Path(args.head_policy).parent / "vecnormalize.pkl"
    head_vn = None
    if head_vn_path.exists():
        dummy_head = DummyVecEnv([lambda: DUMHeadTrackingEnv()])
        head_vn = VecNormalize.load(str(head_vn_path), dummy_head)
        head_vn.training = False
        head_vn.norm_reward = False
        print(f"[setup] head VecNormalize cargado")
    else:
        print(f"[setup] sin VecNormalize del head (no encontrado en {head_vn_path})")
    head_model = PPO.load(args.head_policy)

    # Cargar VecNormalize del ARM
    print("[setup] Cargando arm policy v7e...")
    dummy_arm = DummyVecEnv([lambda: CombinedHeadArmEnv(head_model, head_vn, phase=1)])
    arm_vn_path = Path(args.arm_policy).parent / "vecnormalize.pkl"
    arm_vn = VecNormalize.load(str(arm_vn_path), dummy_arm)
    arm_vn.training = False
    arm_vn.norm_reward = False
    arm_model = PPO.load(args.arm_policy, env=arm_vn)

    raw = arm_vn.venv.envs[0]
    print(f"[setup] env OK. nu={raw.model.nu} dof={raw.model.nv}")
    if args.enable_throw:
        raw.set_throw_enabled(True)
        print("[setup] throw ENABLED (HELD->THROWN post-grab)")

    # Renderer offscreen
    renderer = mujoco.Renderer(raw.model, height=480, width=480)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # v16: cap 35/15. Solo falling (no static), distribucion de training.
    configs = [
        (0.00, 0.00, 0.0,  "rehearsal off=0"),
        (0.15, 0.05, -2.5, "falling small (15, 5)"),
        (0.20, 0.08, -2.5, "falling small (20, 8)"),
        (0.25, 0.10, -2.5, "falling mid (25, 10)"),
        (0.30, 0.12, -2.5, "falling high (30, 12)"),
        (0.35, 0.10, -2.5, "falling MAX-up (35, 10)"),
        (0.30, 0.15, -2.5, "falling MAX-out (30, 15)"),
        (0.35, 0.15, -2.5, "falling MAX (35, 15) #1"),
        (0.35, 0.15, -2.5, "falling MAX (35, 15) #2"),
        (0.35, 0.15, -2.5, "falling MAX (35, 15) #3"),
        (0.33, 0.14, -2.5, "falling near-max (33, 14)"),
        (0.30, 0.10, -2.5, "falling mid-confort (30, 10)"),
    ]

    if len(configs) < args.episodes:
        configs = configs + [configs[-1]] * (args.episodes - len(configs))
    configs = configs[:args.episodes]

    frames = []
    grabs = 0
    rews = []
    lens = []
    head_thetas = []
    print(f"\n{'ep':>3s} {'config':>50s}  {'rew':>8s} {'len':>4s} {'grab':>5s} {'min_d':>6s} {'head_deg(°)':>10s}")
    print("-" * 100)
    for ep, (up_off, out_off, gravity, label) in enumerate(configs):
        # Resetear
        raw.set_curriculum_max_offsets(0.0, 0.0)
        raw.set_falling_active(False)
        obs = arm_vn.reset()
        # Forzar spawn segun config
        raw.model.opt.gravity[:] = (0.0, 0.0, gravity)
        ee = raw.data.site_xpos[raw._ee_site_id]
        sp = np.array([ee[0], ee[1] - out_off, ee[2] + 0.03 + up_off])
        raw.data.qpos[raw._ball_qpos_adr:raw._ball_qpos_adr+3] = sp
        raw.data.qvel[raw._ball_dof_adr:raw._ball_dof_adr+6] = 0.0
        mujoco.mj_forward(raw.model, raw.data)
        raw._ball_spawn_pos = sp
        raw._curriculum_up_offset = up_off
        raw._curriculum_out_offset = out_off
        raw._target_world_prev = sp.copy()
        raw._prev_head_action = np.zeros(3, dtype=np.float32)

        done = False
        ep_rew = 0.0
        ep_len = 0
        grabbed = False
        min_dist = 1.0
        theta_acc = []
        while not done:
            action, _ = arm_model.predict(obs, deterministic=True)
            obs, r, dones, infos = arm_vn.step(action)
            info = infos[0]
            ep_rew += float(r[0])
            ep_len += 1
            min_dist = min(min_dist, info['ee_ball_dist'])
            if info.get('grabbed_now', False):
                grabbed = True
            theta_acc.append(np.rad2deg(raw._head_theta_log))

            # Render
            renderer.update_scene(raw.data, camera=-1)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            ball_z = raw.data.qpos[raw._ball_qpos_adr+2]
            for i, txt in enumerate([
                "v13 head + v7e arm (combined)",
                label,
                f"ep {ep}  step {ep_len:3d}  arm={raw._side}",
                f"ball_z {ball_z*100:.1f}cm",
                f"dist {info['ee_ball_dist']*100:.2f}cm",
                f"lever {info['lever_q']*1000:.2f}mm",
                f"head_deg {np.rad2deg(raw._head_theta_log):5.1f}°",
                f"rew {ep_rew:+.1f}",
                "GRAB!" if grabbed else "...",
            ]):
                color = (100, 255, 100) if grabbed else (255, 240, 100)
                d.text((6, 6+i*14), txt, fill=color, font=font_small)
            frames.append(np.array(img))

            done = bool(dones[0])

        rews.append(ep_rew)
        lens.append(ep_len)
        head_thetas.append(np.mean(theta_acc) if theta_acc else 0.0)
        if grabbed:
            grabs += 1
        print(f"{ep:>3d} {label[:50]:>50s}  {ep_rew:+8.1f} {ep_len:>4d} {'YES' if grabbed else ' no':>5s} "
              f"{min_dist*100:>6.2f} {np.mean(theta_acc) if theta_acc else 0.0:>10.1f}",
              flush=True)

    print()
    print(f"TOTAL: grabs={grabs}/{args.episodes}  ep_rew={np.mean(rews):+.1f}  "
          f"ep_len={np.mean(lens):.1f}  avg_head_deg={np.mean(head_thetas):.1f}°")

    # Save video
    import imageio
    out_path = Path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=50, quality=8)
    print(f"\nVideo: {out_path}  ({len(frames)} frames @ 50fps)")


if __name__ == "__main__":
    main()
