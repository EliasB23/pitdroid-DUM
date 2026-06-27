"""Evaluacion de una policy PPO entrenada. Genera estadisticas y opcionalmente video.

Uso:
    python eval_ppo.py runs/ppo_dum_v8_tuned/final.zip
    python eval_ppo.py runs/ppo_dum_v8_tuned/final.zip --video out.mp4 --episodes 3
    python eval_ppo.py runs/ppo_dum_v8_tuned/final.zip --video  # autogenera eval_v8.mp4
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco
from stable_baselines3 import PPO
from rl_env import DUMHeadTrackingEnv


def evaluate(model, env, n_episodes=10, deterministic=True):
    """Corre n_episodes y devuelve estadisticas agregadas."""
    rewards, lengths = [], []
    final_thetas = []
    focus_fires = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, trunc = False, False
        ep_rew = 0.0
        ep_len = 0
        thetas = []
        focused = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, trunc, info = env.step(action)
            ep_rew += r
            ep_len += 1
            thetas.append(info["theta_deg"])
            if info["focus_triggered"]:
                focused = True
        rewards.append(ep_rew)
        lengths.append(ep_len)
        final_thetas.append(thetas[-1])
        if focused:
            focus_fires += 1
        print(f"ep {ep:2d}: rew={ep_rew:+7.2f}  len={ep_len}  "
              f"theta_min={min(thetas):.2f} deg  theta_final={thetas[-1]:.2f} deg  "
              f"focus={'YES' if focused else 'no'}")
    print()
    print(f"Sobre {n_episodes} episodios:")
    print(f"  ep_rew_mean = {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    print(f"  theta_final mean = {np.mean(final_thetas):.2f} deg ± {np.std(final_thetas):.2f}")
    print(f"  focus_triggered en {focus_fires}/{n_episodes} episodios")
    return dict(rewards=rewards, final_thetas=final_thetas, focus_fires=focus_fires)


def record_video(model, env, out_path, n_episodes=2, fps=50, model_path=""):
    """Renderiza n_episodios con overlay informativo y los guarda en un mp4.

    El control corre a 50 Hz (timestep=0.005 * frame_skip=4).
    Si se guarda a 50 fps, el video es 1x real-time (asumiendo que el reproductor honra esa fps).
    """
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    # Tamaño limitado al offscreen framebuffer (default 480x480 si no se especifica en el XML
    # vía <visual><global offwidth=... offheight=.../></visual>)
    renderer = mujoco.Renderer(env.model, height=480, width=480)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    model_name = Path(model_path).name if model_path else "unknown"
    try:
        total_steps = int(model.num_timesteps)
    except Exception:
        total_steps = "?"

    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done, trunc = False, False
        step_idx = 0
        ep_reward = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            ep_reward += r
            step_idx += 1

            renderer.update_scene(env.data, camera=-1)
            frame = renderer.render()
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            # Header (sup izq) — info del modelo
            header_lines = [
                f"model:   {model_name}",
                f"trained: {total_steps:,} steps",
                f"speed:   1x real-time @ {fps} fps",
            ]
            y = 6
            for line in header_lines:
                draw.text((6, y), line, fill=(230, 230, 230), font=font_small)
                y += 13

            # Estado del episodio (sup der)
            focus_str = "YES" if info["focus_triggered"] else "no"
            mode_str = info.get("target_mode", "?")
            ep_lines = [
                f"ep {ep}  step {step_idx:3d}/500",
                f"target: {mode_str}",
                f"theta = {info['theta_deg']:5.1f} deg",
                f"ep_reward = {ep_reward:+7.2f}",
                f"focus: {focus_str}",
            ]
            y = 6
            for line in ep_lines:
                tw = draw.textlength(line, font=font)
                draw.text((img.width - tw - 6, y), line, fill=(255, 240, 100), font=font)
                y += 14

            frames.append(np.array(img))

    imageio.mimsave(out_path, frames, fps=fps, quality=8)
    print(f"Video guardado en {out_path}  ({len(frames)} frames @ {fps} fps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", type=str)
    p.add_argument("--episodes", type=int, default=15)
    p.add_argument("--video", type=str, nargs='?', const='AUTO', default=None,
                   help="Si se pasa '--video' sin valor, autogenera eval_v<N>.mp4 "
                        "extrayendo el N del nombre del run. Si se pasa con ruta, la usa tal cual.")
    p.add_argument("--stochastic", action="store_true")
    args = p.parse_args()

    # Resolver --video AUTO -> eval_v<N>.mp4 en la carpeta del run
    if args.video == 'AUTO':
        run_dir = Path(args.model_path).parent
        m = re.search(r'v(\d+)', run_dir.name)
        num = m.group(1) if m else "x"
        args.video = str(run_dir / f"eval_v{num}.mp4")
        print(f"Video auto-naming: {args.video}")

    env = DUMHeadTrackingEnv()
    model = PPO.load(args.model_path)
    print(f"Loaded: {args.model_path}")
    print()

    evaluate(model, env, n_episodes=args.episodes, deterministic=not args.stochastic)

    if args.video:
        print()
        record_video(model, env, args.video,
                     n_episodes=min(3, args.episodes),
                     model_path=args.model_path)


if __name__ == "__main__":
    main()
