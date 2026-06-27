"""Evaluacion del DUMGrabEnv con video MP4 + overlay.

Uso:
    python Scripts/rl/eval_grab.py runs/grab_phase1/final.zip --phase 1 --episodes 6 --video
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "envs"))

import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from grab_env import DUMGrabEnv


def make_env_fn(phase):
    def _f():
        return DUMGrabEnv(phase=phase)
    return _f


def load_with_vecnorm(model_path: Path, phase: int):
    """Carga model + VecNormalize (si existe) y devuelve (model, env)."""
    env_raw = DummyVecEnv([make_env_fn(phase)])
    vn_path = model_path.parent / "vecnormalize.pkl"
    if vn_path.exists():
        env = VecNormalize.load(str(vn_path), env_raw)
        env.training = False
        env.norm_reward = False
        print(f"[eval] cargado VecNormalize desde {vn_path}")
    else:
        env = env_raw
        print(f"[eval] sin VecNormalize (no encontrado en {vn_path})")
    model = PPO.load(str(model_path), env=env)
    return model, env


def evaluate(model_path: Path, phase: int, n_episodes: int = 10, video_path: Path = None):
    model, vec_env = load_with_vecnorm(model_path, phase)
    raw_env = vec_env.venv.envs[0] if hasattr(vec_env, "venv") else vec_env.envs[0]

    frames = []
    if video_path is not None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("PIL no disponible — sin video"); video_path = None
    if video_path is not None:
        renderer = mujoco.Renderer(raw_env.model, height=480, width=480)
        try:
            font = ImageFont.truetype("arial.ttf", 13)
            font_small = ImageFont.truetype("arial.ttf", 11)
        except Exception:
            font = ImageFont.load_default(); font_small = font

    model_name = model_path.parent.name
    try:
        total_steps = int(model.num_timesteps)
    except Exception:
        total_steps = "?"

    rewards, lens, grabs, throws, fails = [], [], [], [], []
    for ep in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_rew = 0.0
        ep_len = 0
        grabbed = False
        thrown = False
        floor_fail = False
        max_grab_z = None
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = vec_env.step(action)
            ep_rew += float(r[0])
            ep_len += 1
            info = infos[0]
            if info.get("grabbed_now", False):
                grabbed = True
                max_grab_z = info.get("grab_height", 0.0)
            if info.get("released_now", False):
                thrown = True
            if info.get("r_floor_fail", 0.0) < 0 and not info.get("was_held", False):
                floor_fail = True
            done = bool(dones[0])

            if video_path is not None:
                renderer.update_scene(raw_env.data, camera=-1)
                frame = renderer.render()
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                # header
                for i, line in enumerate([
                    f"model: {model_name}",
                    f"trained: {total_steps:,} steps",
                    f"phase: {phase}  |  1x @ 50 fps",
                ]):
                    draw.text((6, 6 + i*13), line, fill=(230,230,230), font=font_small)
                # estado del episodio
                state = info.get("state", "?")
                ball_z = info.get("ball_z", 0.0)
                dist = info.get("ee_ball_dist", 0.0)
                lever_q = info.get("lever_q", 0.0)
                side = getattr(raw_env, "_side", "?")
                lines = [
                    f"ep {ep}  step {ep_len:3d}  arm={side}",
                    f"state: {state}",
                    f"ball_z: {ball_z:.3f} m",
                    f"EE-ball: {dist*100:.1f} cm",
                    f"lever:  {lever_q*1000:.1f} mm",
                    f"reward: {ep_rew:+.1f}",
                ]
                for i, line in enumerate(lines):
                    tw = draw.textlength(line, font=font)
                    color = (255, 240, 100) if state != "HELD" else (100, 255, 100)
                    draw.text((img.width - tw - 6, 6 + i*14), line, fill=color, font=font)
                frames.append(np.array(img))

        rewards.append(ep_rew); lens.append(ep_len)
        if grabbed: grabs.append((ep, max_grab_z))
        if thrown: throws.append(ep)
        if floor_fail: fails.append(ep)
        flags = []
        if grabbed: flags.append("GRAB")
        if thrown: flags.append("THROW")
        if floor_fail: flags.append("FAIL")
        print(f"ep {ep:2d}: rew={ep_rew:+8.2f} len={ep_len:3d}  {' '.join(flags) or '(no event)'}")

    print()
    print(f"Sobre {n_episodes} episodios:")
    print(f"  ep_rew_mean = {np.mean(rewards):+.2f} +/- {np.std(rewards):.2f}")
    print(f"  ep_len_mean = {np.mean(lens):.1f}")
    print(f"  grabbed:     {len(grabs)}/{n_episodes}")
    print(f"  thrown:      {len(throws)}/{n_episodes}")
    print(f"  floor fail:  {len(fails)}/{n_episodes}")
    if grabs:
        zs = [g[1] for g in grabs]
        print(f"  grab heights: min={min(zs):.3f} max={max(zs):.3f} mean={np.mean(zs):.3f}")

    if video_path is not None and frames:
        import imageio
        imageio.mimsave(str(video_path), frames, fps=50, quality=8)
        print(f"\nVideo: {video_path}  ({len(frames)} frames @ 50 fps)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", type=str)
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--video", type=str, nargs='?', const='AUTO', default=None,
                   help="--video sin valor autogenera eval_grab_phase<N>.mp4")
    args = p.parse_args()

    mp = Path(args.model_path)
    if args.video == 'AUTO':
        args.video = str(mp.parent / f"eval_grab_phase{args.phase}.mp4")
        print(f"Video auto: {args.video}")

    evaluate(mp, phase=args.phase, n_episodes=args.episodes,
             video_path=Path(args.video) if args.video else None)


if __name__ == "__main__":
    main()
