"""Eval custom para v14_head: testea targets en azimuth EXTREMO (incluye atras).

Por que: eval_ppo.py usa el cono default del env (TARGET_CONE_AZIMUTH=75°),
asi que NUNCA muestrea targets atras del robot. v14_head fue entrenada con cono
extendido a ±150° — necesitamos targets en ese rango para validar el cambio.

Genera 8 episodios con azimuth explicito (sweep -150°→+150°), renderiza video
con overlay mostrando target azimuth y angulo theta del head.

Uso:
    python Scripts/eval_v14_head.py
        [--model runs/ppo_dum_v14_head/final.zip]
        [--video runs/ppo_dum_v14_head/eval_behind.mp4]
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

from rl_env import DUMHeadTrackingEnv, TARGET_DISTANCE_MIN, TARGET_DISTANCE_MAX


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="runs/ppo_dum_v14_head/final.zip")
    p.add_argument("--video", type=str, default="runs/ppo_dum_v14_head/eval_v14_behind.mp4")
    p.add_argument("--steps-per-target", type=int, default=400,
                   help="Steps por target (8s a 50Hz). 400 = suficiente para que HeadRot llegue al limite (6.6s).")
    args = p.parse_args()

    # Lista explicita de target azimuth a probar (en grados)
    target_azs = [0, 45, 90, 120, 150, -90, -120, -150]
    target_el = 10  # elevation moderada para todos

    print(f"[setup] cargando {args.model}")
    model = PPO.load(args.model)

    env = DUMHeadTrackingEnv()
    obs, _ = env.reset(seed=42)
    print(f"[setup] env OK. obs.shape={obs.shape}")

    renderer = mujoco.Renderer(env.model, height=480, width=480)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    frames = []
    summary = []

    def target_world_for(az_deg, el_deg, dist=0.6):
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        # Replica de _sample_point_in_cone con valores fijos
        target_dir = env._rot_z(az) @ env._rot_x(el) @ env._forward_world_init
        target_dir /= np.linalg.norm(target_dir) + 1e-9
        return env._head_pos_init + dist * target_dir

    for az_deg in target_azs:
        # Reset env y forzar target estatico
        obs, _ = env.reset(seed=int(abs(az_deg) * 10) + 1)
        target_world = target_world_for(az_deg, target_el)
        env._target_mode = "static"
        env._target_a = target_world
        env._target_world = target_world.copy()
        env.data.mocap_pos[env._target_mocap_idx] = target_world
        mujoco.mj_forward(env.model, env.data)

        # Trackeo
        theta_history = []
        head_rot_history = []
        head_base_history = []
        for step in range(args.steps_per_target):
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)

            theta = env._angle_to_target()
            theta_history.append(theta)
            head_rot_history.append(float(env.data.qpos[env._head_qpos_adr[2]]))  # HeadRotation_joint
            head_base_history.append(float(env.data.qpos[env._head_qpos_adr[1]]))  # HeadBase_joint

            # Render
            renderer.update_scene(env.data, camera=-1)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            head_rot_deg = np.rad2deg(env.data.qpos[env._head_qpos_adr[2]])
            head_base_deg = np.rad2deg(env.data.qpos[env._head_qpos_adr[1]])
            lines = [
                "v14_head — eval con targets extremos",
                f"target az={az_deg:+4d}°  el={target_el}°",
                f"step {step:3d}",
                f"theta_to_target: {np.rad2deg(theta):5.1f}°",
                f"HeadRot (yaw):   {head_rot_deg:+6.1f}°",
                f"HeadBase(pitch): {head_base_deg:+6.1f}°",
            ]
            for i, ln in enumerate(lines):
                color = (100, 255, 100) if (np.rad2deg(theta) < 10) else (255, 240, 100)
                d.text((6, 6 + i * 14), ln, fill=color, font=font_small)
            frames.append(np.array(img))

        # Stats finales (ultimo segundo del episodio = settling)
        last_50 = theta_history[-50:]
        last_50_rot = head_rot_history[-50:]
        last_50_base = head_base_history[-50:]
        theta_final = float(np.mean(last_50))
        rot_final = float(np.mean(last_50_rot))
        base_final = float(np.mean(last_50_base))
        summary.append({
            "az": az_deg,
            "theta_deg_final": np.rad2deg(theta_final),
            "head_rot_deg_final": np.rad2deg(rot_final),
            "head_base_deg_final": np.rad2deg(base_final),
        })
        print(f"  az={az_deg:+4d}°: theta_final={np.rad2deg(theta_final):5.1f}°  "
              f"HeadRot={np.rad2deg(rot_final):+6.1f}°  HeadBase={np.rad2deg(base_final):+6.1f}°")

    # Video
    import imageio
    out_path = Path(args.video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=50, quality=8)
    print(f"\nVideo: {out_path}  ({len(frames)} frames @ 50fps)")

    # Diagnostico
    print("\n=== DIAGNOSTICO ===")
    for s in summary:
        # Para tracking correcto: HeadRot deberia ser cercano al azimuth del target
        # HeadBase deberia ser cercano a el target_el (no perpendicular al piso)
        rot_ok = abs(s["head_rot_deg_final"] - s["az"]) < 30
        head_stable = abs(s["head_base_deg_final"]) < 40  # no tilteado al piso
        flag = "OK" if (s["theta_deg_final"] < 15 and rot_ok and head_stable) else "FAIL"
        print(f"  az={s['az']:+4d}°  theta={s['theta_deg_final']:5.1f}°  "
              f"rot={s['head_rot_deg_final']:+6.1f}°  base={s['head_base_deg_final']:+6.1f}°  [{flag}]")


if __name__ == "__main__":
    main()
