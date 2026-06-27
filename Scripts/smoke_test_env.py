"""Smoke test del DUMHeadTrackingEnv.
Verifica: carga del XML, shapes de espacios, reset, 10 steps con accion aleatoria,
trigger del hook de enfoque con un controlador trivial que apunta al target."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rl_env import DUMHeadTrackingEnv

print("=== Carga ===")
env = DUMHeadTrackingEnv()
print(f"obs_space: {env.observation_space}")
print(f"act_space: {env.action_space}")
print(f"head actuators ids: {env._head_act_ids}")
print(f"head ctrlrange lo: {env._head_ctrl_lo}")
print(f"head ctrlrange hi: {env._head_ctrl_hi}")
print(f"lens max: {env._lens_max}")

print("\n=== Reset ===")
obs, info = env.reset(seed=42)
print(f"obs shape: {obs.shape}, obs[:6] (qpos,qvel head): {obs[:6]}")
print(f"target dir local: {obs[6:9]}, target dist: {obs[9]:.3f}")
print(f"target world: {env._target_world}")
print(f"head world pos: {env.data.xpos[env._head_body_id]}")
print(f"theta inicial: {np.rad2deg(env._angle_to_target()):.2f} deg")

print("\n=== 5 steps con accion random ===")
rng = np.random.default_rng(0)
for i in range(5):
    a = rng.uniform(-1, 1, size=3).astype(np.float32)
    obs, r, term, trunc, info = env.step(a)
    print(f"step {i}: reward={r:+.4f}  theta={info['theta_deg']:6.2f}°  "
          f"r_track={info['r_track']:+.3f}  r_smooth={info['r_smooth']:+.4f}  "
          f"r_effort={info['r_effort']:+.4f}  focus={info['focus_triggered']}")

print("\n=== 200 steps con controlador 'mira al target' (P simple) ===")
# Controlador: cada head joint hacia donde acerca el forward al target.
# No es preciso pero deberia disparar el hook de enfoque en algun momento.
env.reset(seed=42)
focus_step = None
for i in range(200):
    # Heuristica: empujar joints hacia la direccion target en frame local
    tdir = env._target_local()
    tdir = tdir / (np.linalg.norm(tdir) + 1e-9)
    # tdir.y ~ -1 cuando esta en target (forward = -y local)
    # mapeo crudo: usar componentes para deducir señal de cada joint
    a = np.array([
        -tdir[2] * 2.0,   # Neck (axis x) -> pitch arriba/abajo -> sensible a z local
        tdir[2] * 2.0,    # HeadBase (axis x) -> similar pitch
        -tdir[0] * 2.0,   # HeadRot (axis z) -> yaw -> sensible a x local
    ], dtype=np.float32)
    a = np.clip(a, -1, 1)
    obs, r, term, trunc, info = env.step(a)
    if info['focus_triggered'] and focus_step is None:
        focus_step = i
        print(f"FOCUS triggered at step {i}, theta={info['theta_deg']:.2f}°")
    if i % 50 == 0 or i == 199:
        print(f"step {i:3d}: theta={info['theta_deg']:6.2f}°  r={r:+.3f}  focus_anim={info['focus_anim_step']}")

print(f"\nfocus_triggered final: {env._focus_triggered}, anim_step final: {env._focus_anim_step}")

print("\n=== Test final: 600 steps -> truncated ===")
env.reset(seed=1)
trunc = False
n = 0
while not trunc:
    obs, r, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    n += 1
print(f"truncated despues de {n} steps (esperado: 500)")

print("\n=== OK ===")
