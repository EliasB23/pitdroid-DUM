"""Motor de animacion DUM4 — runtime integrado.

Combina:
- v13 head policy: trackea el target activo (red ball o yellow ball segun estado)
- v12/v13 arm policy: realiza grab + hold + throw cuando se dispara
- Web server: slider para mover el red mocap + boton "agarrar bola amarilla"
- State machine simple: IDLE <-> GRAB_CYCLE

Flow:
    IDLE (default):
        - red ball mocap movido por slider web
        - head policy: target = red ball pos
        - arm: ctrl neutro (no policy)
        - yellow ball: oculta (z = -10)

    GRAB_CYCLE (al apretar boton):
        1. spawn yellow_ball arriba de la palma (palm_z + 0.40)
        2. head policy: target = yellow_ball pos
        3. arm policy: activa
        4. env state machine (interno): FALLING -> HELD -> THROWN
        5. en THROWN, aplicar xfrc para reducir gravedad efectiva ~80%
           -> bola flota 2-3 segundos
        6. al touchdown o timeout 3s post-release: volver a IDLE
        7. head policy vuelve a target = red ball

Uso:
    python Scripts/run_animation_engine.py
        [--arm-policy runs/grab_phase1_v13_natural/final.zip]
        [--head-policy runs/ppo_dum_v13/final.zip]
        [--host 0.0.0.0 --port 8000]

Web: http://localhost:8000
"""
from __future__ import annotations

import argparse
import io
import queue
import sys
import threading
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env import DUMHeadTrackingEnv
from rl.envs.grab_env import (
    DUMGrabEnv, PALM_REST_WORLD, INIT_BALL_ABOVE_PALM,
    THROW_HELD_MAX_S, B_GRAB_SIMPLE, THROW_GRAB_BONUS,
)
from rl.procedural.wave import WaveAnimation
from skins import apply_skin, DEFAULT_SKIN
import web_remote.server as server


# ---------- Estados del motor ----------

STATE_IDLE = "IDLE"
STATE_SPAWN_AND_FALL = "SPAWN_AND_FALL"
STATE_HELD = "HELD"
STATE_THROWN = "THROWN"
STATE_FOLLOW = "FOLLOW"  # 3s post-release siguiendo la bola con cabeza
STATE_WAVING = "WAVING"  # saludo procedural (4s)


# Autofoco animatronico (replica del DUMHeadTrackingEnv en rl_env.py)
FOCUS_THRESHOLD = 0.07         # rad ≈ 4° — bajo de esto se dispara el foco
FOCUS_SETTLING_STEPS = 10
FOCUS_ANIM_STEPS = 50          # ~1s a 50Hz
FOCUS_REARM_LOST_STEPS = 30    # 0.6s out-of-focus para rearmar
FOCUS_REARM_THRESHOLD_MULT = 3.0
FOCUS_N_CYCLES_RANGE = (1, 4)
FOCUS_AMPLITUDE_RANGE = (0.5, 1.0)
FOCUS_RESTING_RANGE = (0.0, 0.15)
HEAD_FORWARD_LOCAL = np.array([0.0, -1.0, 0.0])


# ---------- Constantes ----------

YELLOW_BALL_HIDDEN_Z = -10.0   # cuando no se usa, la mandamos lejos
GRAB_CYCLE_SPAWN_ABOVE = 0.40  # spawn yellow_ball a palm + 0.40m
FOLLOW_DURATION_S = 3.0         # segundos post-throw siguiendo la bola
GRAVITY_CANCEL_FACTOR = 0.80   # durante THROWN, cancelamos 80% de la gravedad
SIM_HZ = 50
FRAME_SKIP = 4
TIMESTEP = 0.005

# NO MAS THROW BOOST: la velocidad del throw sale 100% de la policy.
# Si la bola no llega lejos, mas training (v14+).


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm-policy", type=str,
                   default="runs/grab_phase1_v16_extended/final.zip",
                   help="Arm policy zip. Default v16_extended (cap 35/15). "
                        "Si querés volver al v14_long (mejor catch global, cap 40/10), "
                        "pasar --arm-policy runs/grab_phase1_v14_long/final.zip")
    p.add_argument("--head-policy", type=str,
                   default="runs/ppo_dum_v14c_aggressive/final.zip",
                   help="Head policy zip. Default v14c (usa HeadRot al limite fisico ±160°). "
                        "v13 original tenia cono ±75° y compensaba con HeadBase tilt.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-viewer", action="store_true",
                   help="No abrir viewer interactivo (util para headless con solo web stream)")
    args = p.parse_args()

    # Resolver rutas de policy como ABSOLUTAS respecto a la raiz del proyecto,
    # asi el motor funciona sin importar el cwd desde donde se lo lance
    # (ej. run_demo.ps1, doble-click, otra carpeta).
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    def _resolve_policy(pstr):
        pp = Path(pstr)
        return pp if pp.is_absolute() else (PROJECT_ROOT / pp)

    # Fallback chain: v16 -> v14_long -> v13 -> error
    arm_path = _resolve_policy(args.arm_policy)
    if not arm_path.exists():
        fallback_chain = [
            SCRIPTS_DIR.parent / "runs" / "grab_phase1_v14_long" / "final.zip",
            SCRIPTS_DIR.parent / "runs" / "grab_phase1_v13_natural" / "final.zip",
        ]
        for fallback in fallback_chain:
            if fallback.exists():
                arm_path = fallback
                print(f"[setup] arm policy default no existe, fallback a {arm_path.parent.name}")
                break
        else:
            raise FileNotFoundError(f"No arm policy: {args.arm_policy} ni fallbacks")
    print(f"[setup] arm policy: {arm_path}")

    # --- Cargar arm policy + VecNormalize ---
    dummy_arm = DummyVecEnv([lambda: DUMGrabEnv(phase=1)])
    arm_vn_path = arm_path.parent / "vecnormalize.pkl"
    arm_vn = VecNormalize.load(str(arm_vn_path), dummy_arm)
    arm_vn.training = False
    arm_vn.norm_reward = False
    arm_model = PPO.load(str(arm_path), env=arm_vn)
    arm_env = arm_vn.venv.envs[0]
    # Activar throw permanentemente (el state machine externo controla cuando se "usa")
    arm_env.set_throw_enabled(True)
    arm_env.set_falling_active(True)
    arm_env.set_subphase(0)

    # --- Cargar head policy (v13 sin VecNormalize) ---
    head_path = _resolve_policy(args.head_policy)
    head_model = PPO.load(str(head_path))
    print(f"[setup] head policy: {head_path}")

    model = arm_env.model
    data = arm_env.data

    # Skin inicial del robot (cambiable en runtime desde la web)
    apply_skin(model, DEFAULT_SKIN)
    print(f"[setup] skin inicial: {DEFAULT_SKIN}")

    # --- IDs adicionales para el head obs ---
    head_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LenteExt_link")
    lens_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lens_center")
    head_joint_names = ["Neck_joint", "HeadBase_joint", "HeadRotation_joint"]
    head_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in head_joint_names]
    head_qpos_adr = np.array([model.jnt_qposadr[j] for j in head_joint_ids])
    head_dof_adr = np.array([model.jnt_dofadr[j] for j in head_joint_ids])
    hb_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "HipBody_joint")
    hb_qpos_adr = model.jnt_qposadr[hb_jid]
    hb_dof_adr = model.jnt_dofadr[hb_jid]
    head_act_names = ["act_Neck", "act_HeadBase", "act_HeadRot"]
    head_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in head_act_names]
    head_ctrl_lo = np.array([model.actuator_ctrlrange[a, 0] for a in head_act_ids])
    head_ctrl_hi = np.array([model.actuator_ctrlrange[a, 1] for a in head_act_ids])
    # Lente actuator para autofoco
    lens_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_LenteExt")
    lens_ctrl_lo, lens_ctrl_hi = model.actuator_ctrlrange[lens_act_id]
    LENS_MAX = float(lens_ctrl_hi)  # ~0.021
    # Red ball mocap (target visual del slider web)
    target_mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_idx = model.body_mocapid[target_mocap_body]

    # --- Colas y servidor web ---
    target_q: queue.Queue = queue.Queue(maxsize=1)
    telemetry_q: queue.Queue = queue.Queue(maxsize=1)
    frame_q: queue.Queue = queue.Queue(maxsize=1)
    event_q: queue.Queue = queue.Queue(maxsize=8)
    server.init_queues(target_q, telemetry_q, frame_q, event_q)

    def run_uvicorn():
        import uvicorn
        config = uvicorn.Config(server.app, host=args.host, port=args.port, log_level="warning")
        srv = uvicorn.Server(config)
        srv.run()

    th = threading.Thread(target=run_uvicorn, daemon=True, name="uvicorn")
    th.start()
    print(f"[server] http://{args.host}:{args.port}")
    print(f"[server] local: http://127.0.0.1:{args.port}")

    # --- Estado del runtime ---
    sm_state = STATE_IDLE
    red_target_world = np.array([-0.012, -0.50, 0.50])  # default red mocap pos
    yellow_hidden_pos = np.array([0.0, 0.0, YELLOW_BALL_HIDDEN_Z])
    target_world_prev = red_target_world.copy()
    prev_head_action = np.zeros(3, dtype=np.float32)
    cycle_start_t = 0.0
    follow_start_t = None
    grav_default = float(arm_env._current_falling_gravity)

    # --- Estado del autofoco (replica de DUMHeadTrackingEnv) ---
    focus_triggered = False
    focus_anim_step = -1
    focus_amplitude = 0.7
    focus_n_cycles = 2
    focus_resting = 0.0
    focus_lost_steps = 0
    focus_settling_counter = 0  # cuenta steps con head settled (para SETTLING_STEPS)

    # --- Wave (saludo) ---
    wave_left = WaveAnimation(side="left", model=model)
    wave_right = WaveAnimation(side="right", model=model)
    wave_current = None  # apunta a wave_left o wave_right cuando activo
    wave_start_t = None

    # Renderer offscreen para stream MJPEG. Usamos directamente 480x480 (lo que el
    # framebuffer del modelo soporta). CSS object-fit:cover llena el panel.
    # Stream MJPEG optimizado para datos moviles: menos resolucion, menos fps y
    # menos calidad JPEG = menos ancho de banda (menos "trabado" con datos).
    # 400x400 @ ~16 fps q58 ~ 2 Mbps (vs 480 @ 25fps q75 ~ 6-8 Mbps).
    STREAM_SIZE = 400
    STREAM_QUALITY = 58
    stream_renderer = mujoco.Renderer(model, height=STREAM_SIZE, width=STREAM_SIZE)
    print(f"[setup] renderer {STREAM_SIZE}x{STREAM_SIZE} (CSS object-fit:cover lo escala al panel)")
    frame_every_n = 3          # 1 frame cada 3 steps de control (50Hz) -> ~16.7 fps
    frame_counter = 0

    # --- Camara orbital del stream (D-pad desde la web) ---
    # Siempre centrada en el robot (lookat fijo) a distancia fija -> orbita en una
    # esfera. Izq/der giran (azimuth); arriba/abajo cambian la ALTURA (elevation),
    # con un piso: la camara no baja de la altura actual (solo puede subir).
    CAM_LOOKAT = (0.07, 0.07, 0.32)
    CAM_DISTANCE = 1.5
    CAM_AZ_STEP = 30.0     # grados por click izq/der
    CAM_EL_STEP = 15.0     # grados por click arriba/abajo
    CAM_EL_LOW = -12.0     # altura MINIMA (la actual) — la camara no baja de aca
    CAM_EL_HIGH = -85.0    # altura MAXIMA (casi cenital)
    stream_cam = mujoco.MjvCamera()
    stream_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    stream_cam.lookat[:] = CAM_LOOKAT
    stream_cam.distance = CAM_DISTANCE
    stream_cam.azimuth = 90.0       # frente del robot
    stream_cam.elevation = CAM_EL_LOW

    # Hide yellow ball initially
    arm_env.data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3] = yellow_hidden_pos
    arm_env.data.qvel[arm_env._ball_dof_adr:arm_env._ball_dof_adr+6] = 0.0
    mujoco.mj_forward(model, data)

    def hide_yellow_ball():
        data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3] = yellow_hidden_pos
        data.qvel[arm_env._ball_dof_adr:arm_env._ball_dof_adr+6] = 0.0

    def spawn_yellow_ball():
        # Aleatorizar side (R o L)
        arm_env._side = "R" if np.random.uniform() < 0.5 else "L"
        # Randomizar color
        if arm_env._ball_geom_id >= 0:
            rgb = np.random.uniform(0.3, 1.0, size=3)
            arm_env.model.geom_rgba[arm_env._ball_geom_id, :3] = rgb
            arm_env.model.geom_rgba[arm_env._ball_geom_id, 3] = 1.0
        # Spawn arriba de la palma activa
        ee = data.site_xpos[arm_env._ee_site_id]
        spawn = np.array([float(ee[0]), float(ee[1]), float(ee[2]) + GRAB_CYCLE_SPAWN_ABOVE])
        data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3] = spawn
        data.qpos[arm_env._ball_qpos_adr+3:arm_env._ball_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[arm_env._ball_dof_adr:arm_env._ball_dof_adr+6] = 0.0
        # Reset state del env interno
        from rl.envs.grab_env import STATE_FALLING
        arm_env.state = STATE_FALLING
        arm_env._grab_bonus_consumed = False
        arm_env.t_grab = None
        arm_env.t_release = None
        arm_env.was_held = False
        arm_env._sim_t = 0.0
        mujoco.mj_forward(model, data)

    def soft_reset(reason=""):
        """Reinicio SUAVE: deja el robot en un estado IDLE limpio SIN matar el
        proceso ni cortar la web. Lo dispara el boton de la web, o automaticamente
        si la fisica diverge (NaN) o salta una excepcion en el loop."""
        nonlocal sm_state, wave_current, wave_start_t, follow_start_t, cycle_start_t
        nonlocal focus_triggered, focus_anim_step, focus_lost_steps, focus_settling_counter
        nonlocal prev_head_action, target_world_prev
        # cancelar saludo y restaurar boost de actuadores si quedo activo
        if wave_current is not None:
            wave_current.cancel()
            wave_current = None
        wave_start_t = None
        # limpiar fuerzas externas (la cancelacion de gravedad del throw, etc.)
        data.xfrc_applied[:] = 0.0
        # reset fisico completo a la pose neutra (garantiza estado finito, sin NaN)
        mujoco.mj_resetData(model, data)
        # desactivar las connect equality por las dudas
        for s in ("R", "L"):
            eid = arm_env._side_info[s]["connect_eq_id"]
            arm_env.model.eq_active0[eid] = 0
            if hasattr(data, "eq_active"):
                data.eq_active[eid] = 0
        # esconder la bola amarilla
        data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3] = yellow_hidden_pos
        data.qvel[arm_env._ball_dof_adr:arm_env._ball_dof_adr+6] = 0.0
        mujoco.mj_forward(model, data)
        # reset del estado interno del env del brazo
        arm_env.state = "FALLING"
        arm_env._grab_bonus_consumed = False
        arm_env.t_grab = None
        arm_env.t_release = None
        arm_env.was_held = False
        # reset del estado del runtime
        sm_state = STATE_IDLE
        focus_triggered = False
        focus_anim_step = -1
        focus_lost_steps = 0
        focus_settling_counter = 0
        follow_start_t = None
        cycle_start_t = 0.0
        prev_head_action = np.zeros(3, dtype=np.float32)
        target_world_prev = red_target_world.copy()
        tag = f" ({reason})" if reason else ""
        print(f"[runtime] SOFT RESET{tag} -> IDLE", flush=True)

    def compute_head_obs(target_world, prev_action):
        head_mat = data.xmat[head_body].reshape(3, 3)
        lens_origin = data.site_xpos[lens_site]
        tloc = head_mat.T @ (target_world - lens_origin)
        tdist = float(np.linalg.norm(tloc))
        tdir = tloc / (tdist + 1e-9)
        dt_ctrl = FRAME_SKIP * TIMESTEP
        tvel_world = (target_world - target_world_prev) / dt_ctrl
        tvel = head_mat.T @ tvel_world
        qpos_head = data.qpos[head_qpos_adr]
        qvel_head = data.qvel[head_dof_adr]
        qpos_hb = float(data.qpos[hb_qpos_adr])
        qvel_hb = float(data.qvel[hb_dof_adr])
        return np.concatenate([
            qpos_head, qvel_head, [qpos_hb, qvel_hb],
            tdir, [tdist], tvel,
            prev_action.astype(np.float64),
        ])

    def head_action_to_ctrl(action):
        a = np.clip(action, -1.0, 1.0).astype(np.float64)
        return (a + 1.0) * 0.5 * (head_ctrl_hi - head_ctrl_lo) + head_ctrl_lo

    print(f"[runtime] estado inicial: {sm_state}")
    print(f"[runtime] CTRL+C para terminar")

    sim_dt = FRAME_SKIP * TIMESTEP  # 0.02s = 50 Hz de control
    viewer_ctx = None
    if not args.no_viewer:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        print(f"[viewer] abierto. Cerrar el viewer termina el programa.")

    last_log_t = time.perf_counter()
    sim_t_total = 0.0
    last_telemetry_t = 0.0

    try:
        while True:
            loop_start = time.perf_counter()
            if viewer_ctx is not None and not viewer_ctx.is_running():
                break

            # --- 1. Leer eventos del web (botones "agarrar", "saludar") ---
            try:
                ev = event_q.get_nowait()
                ev_type = ev.get("type")
                if ev_type == "grab_yellow" and sm_state == STATE_IDLE:
                    print(f"[runtime] EVENT grab_yellow recibido -> SPAWN_AND_FALL")
                    spawn_yellow_ball()
                    # Reset target_world_prev a la NUEVA posicion de la bola para
                    # evitar tvel_local explosivo (el bola "salto" de z=-10 a palm_z).
                    new_ball_pos = data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3].copy()
                    target_world_prev = new_ball_pos
                    sm_state = STATE_SPAWN_AND_FALL
                    cycle_start_t = sim_t_total
                elif ev_type == "wave" and sm_state == STATE_IDLE:
                    # Randomizar el lado del saludo
                    chosen_side = "right" if np.random.uniform() < 0.5 else "left"
                    wave_current = wave_right if chosen_side == "right" else wave_left
                    wave_current.trigger(sim_t_total)
                    wave_start_t = sim_t_total
                    sm_state = STATE_WAVING
                    print(f"[runtime] EVENT wave recibido -> WAVING ({chosen_side})")
                elif ev_type == "set_skin":
                    # Cambio de apariencia — cosmetico, funciona en cualquier estado
                    skin_name = ev.get("name", "")
                    if apply_skin(model, skin_name):
                        print(f"[runtime] EVENT set_skin -> {skin_name}")
                    else:
                        print(f"[runtime] set_skin desconocido: {skin_name}")
                elif ev_type == "cam":
                    # D-pad de la camara: izq/der giran, arriba/abajo cambian altura.
                    d = ev.get("dir", "")
                    if d == "left":
                        stream_cam.azimuth = (stream_cam.azimuth - CAM_AZ_STEP) % 360
                    elif d == "right":
                        stream_cam.azimuth = (stream_cam.azimuth + CAM_AZ_STEP) % 360
                    elif d == "up":
                        # subir la camara = elevation mas negativa (clamp a CAM_EL_HIGH)
                        stream_cam.elevation = max(CAM_EL_HIGH, stream_cam.elevation - CAM_EL_STEP)
                    elif d == "down":
                        # bajar = elevation menos negativa, pero no por debajo de CAM_EL_LOW
                        stream_cam.elevation = min(CAM_EL_LOW, stream_cam.elevation + CAM_EL_STEP)
                    print(f"[runtime] cam {d} -> az={stream_cam.azimuth:.0f} el={stream_cam.elevation:.0f}")
                elif ev_type == "reset":
                    soft_reset("boton web")
            except queue.Empty:
                pass

            # --- 2. Actualizar red mocap target desde slider web ---
            try:
                td = target_q.queue[0]  # peek
                red_target_world = np.array([float(td["x"]), float(td["y"]), float(td["z"])])
            except (IndexError, queue.Empty):
                pass
            data.mocap_pos[target_mocap_idx] = red_target_world

            # --- 3. Decidir target del head segun estado ---
            ball_pos = data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3].copy()
            # Solo grab cycle (SPAWN_AND_FALL, HELD, THROWN, FOLLOW) usa la bola amarilla
            # como target. WAVING e IDLE siguen la bola roja (slider del usuario).
            if sm_state in (STATE_SPAWN_AND_FALL, STATE_HELD, STATE_THROWN, STATE_FOLLOW):
                head_target = ball_pos.copy()
            else:
                head_target = red_target_world

            # --- 4. Head action ---
            head_obs = compute_head_obs(head_target, prev_head_action)
            head_obs_np = head_obs[None, :].astype(np.float32)
            head_action, _ = head_model.predict(head_obs_np, deterministic=True)
            head_action = head_action[0]
            prev_head_action = head_action.astype(np.float32).copy()
            head_ctrl = head_action_to_ctrl(head_action)

            # --- 5. Arm action ---
            if sm_state in (STATE_SPAWN_AND_FALL, STATE_HELD, STATE_THROWN, STATE_FOLLOW):
                # Activar policy del brazo
                arm_obs = arm_env._get_obs()
                arm_obs_norm = arm_vn.normalize_obs(arm_obs[None, :])
                arm_action_raw, _ = arm_model.predict(arm_obs_norm, deterministic=True)
                arm_action = arm_action_raw[0]
                arm_ctrl = arm_env._action_to_arm_ctrl(arm_action)
            else:
                # IDLE o WAVING: arm policy desactivado
                arm_ctrl = None
                arm_action = np.zeros(5, dtype=np.float32)

            # --- 5b. Autofoco — calcular theta_to_target y update del state machine ---
            head_mat_now = data.xmat[head_body].reshape(3, 3)
            forward_world = head_mat_now @ HEAD_FORWARD_LOCAL
            lens_origin_now = data.site_xpos[lens_site]
            delta = head_target - lens_origin_now
            d_norm = float(np.linalg.norm(delta))
            if d_norm > 1e-9:
                cos_theta = float(np.clip(np.dot(forward_world, delta / d_norm), -1.0, 1.0))
                theta_to_target = float(np.arccos(cos_theta))
            else:
                theta_to_target = 0.0
            # FSM del foco
            if not focus_triggered:
                # Trigger cuando el head se asienta cerca del target
                if theta_to_target < FOCUS_THRESHOLD:
                    focus_settling_counter += 1
                    if focus_settling_counter >= FOCUS_SETTLING_STEPS:
                        focus_triggered = True
                        focus_anim_step = 0
                        focus_n_cycles = int(np.random.randint(*FOCUS_N_CYCLES_RANGE))
                        focus_amplitude = float(np.random.uniform(*FOCUS_AMPLITUDE_RANGE))
                        focus_resting = float(np.random.uniform(*FOCUS_RESTING_RANGE)) * LENS_MAX
                        print(f"[autofoco] TRIGGER (theta={np.rad2deg(theta_to_target):.1f}° "
                              f"cycles={focus_n_cycles} amp={focus_amplitude:.2f})")
                else:
                    focus_settling_counter = 0
            else:
                # Re-arm si el target se aleja sostenidamente
                if theta_to_target > FOCUS_THRESHOLD * FOCUS_REARM_THRESHOLD_MULT:
                    focus_lost_steps += 1
                    if focus_lost_steps >= FOCUS_REARM_LOST_STEPS:
                        focus_triggered = False
                        focus_anim_step = -1
                        focus_lost_steps = 0
                        focus_settling_counter = 0
                        print(f"[autofoco] REARM (theta={np.rad2deg(theta_to_target):.1f}°)")
                else:
                    focus_lost_steps = 0
                if focus_anim_step >= 0:
                    focus_anim_step += 1
            # Computar lens ctrl segun fase de la animacion
            if not focus_triggered or focus_anim_step < 0:
                lens_ctrl = 0.0
            elif focus_anim_step >= FOCUS_ANIM_STEPS:
                lens_ctrl = focus_resting
            else:
                phase = focus_anim_step / FOCUS_ANIM_STEPS
                peak = focus_amplitude * LENS_MAX
                lens_ctrl = peak * 0.5 * (1.0 - np.cos(2.0 * focus_n_cycles * np.pi * phase))

            # --- 6. Construir full_ctrl ---
            full_ctrl = np.zeros(model.nu, dtype=np.float64)
            for aid in range(model.nu):
                lo, hi = model.actuator_ctrlrange[aid]
                full_ctrl[aid] = 0.0 if (lo <= 0.0 <= hi) else 0.5 * (lo + hi)
            # head ctrl
            for i, aid in enumerate(head_act_ids):
                full_ctrl[aid] = head_ctrl[i]
            # lens ctrl (autofoco)
            full_ctrl[lens_act_id] = lens_ctrl
            # arm ctrl si aplica
            if arm_ctrl is not None:
                for i, aid in enumerate(arm_env._arm_act_ids):
                    full_ctrl[aid] = arm_ctrl[i]
            # wave ctrl override (aditivo) — escribe sobre 4 actuadores del brazo elegido
            if sm_state == STATE_WAVING and wave_current is not None:
                wave_ctrl_dict = wave_current.get_ctrl(sim_t_total)
                if wave_ctrl_dict:
                    for act_name, val in wave_ctrl_dict.items():
                        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                        if aid >= 0:
                            full_ctrl[aid] = val

            # --- 7. Gravedad efectiva sobre la bola (cancel parcial durante THROWN/FOLLOW) ---
            if sm_state in (STATE_THROWN, STATE_FOLLOW):
                # Aplicar fuerza hacia arriba sobre la bola que cancela ~80% del peso
                ball_mass = model.body_mass[arm_env._ball_body_id]
                grav = model.opt.gravity[2]
                cancel_force_z = -GRAVITY_CANCEL_FACTOR * ball_mass * grav  # contrario a peso
                data.xfrc_applied[arm_env._ball_body_id, :] = 0.0
                data.xfrc_applied[arm_env._ball_body_id, 2] = cancel_force_z
            else:
                data.xfrc_applied[arm_env._ball_body_id, :] = 0.0

            # --- 8. Step (blindado: si diverge o falla, reset suave en vez de crashear) ---
            data.ctrl[:] = np.nan_to_num(full_ctrl, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                for _ in range(FRAME_SKIP):
                    mujoco.mj_step(model, data)
            except Exception as e:
                soft_reset(f"error en step: {e}")
                continue
            # Deteccion de divergencia de la fisica (NaN/inf en estado)
            if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                soft_reset("fisica diverge (NaN)")
                continue
            sim_t_total += sim_dt

            # En IDLE, pinear la bola al hidden pos (la gravedad la haria caer al infinito)
            if sm_state == STATE_IDLE:
                data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3] = yellow_hidden_pos
                data.qvel[arm_env._ball_dof_adr:arm_env._ball_dof_adr+6] = 0.0
                mujoco.mj_forward(model, data)

            # En HELD, manual override del qpos de la bola (sigue la palma)
            from rl.envs.grab_env import STATE_FALLING as EnvFalling
            if sm_state == STATE_HELD or arm_env.state == "HELD":
                anchor_world = arm_env._ee_pos_world().copy()
                data.qpos[arm_env._ball_qpos_adr:arm_env._ball_qpos_adr+3] = anchor_world
                ee_v = arm_env._ee_linvel_world()
                data.qvel[arm_env._ball_dof_adr:arm_env._ball_dof_adr+3] = ee_v
                data.qvel[arm_env._ball_dof_adr+3:arm_env._ball_dof_adr+6] = 0.0
                mujoco.mj_forward(model, data)

            # --- 9. Transiciones del state machine externo ---
            cycle_dt = sim_t_total - cycle_start_t
            if sm_state == STATE_SPAWN_AND_FALL:
                # Detectar grab por la condicion del env
                if arm_env._detect_grab() and not arm_env._grab_bonus_consumed:
                    arm_env.state = "HELD"
                    arm_env.t_grab = sim_t_total - cycle_start_t
                    arm_env.was_held = True
                    arm_env._grab_bonus_consumed = True
                    sm_state = STATE_HELD
                    print(f"[runtime] GRAB! @ t={cycle_dt:.2f}s -> HELD")
                # Timeout de catch
                elif cycle_dt > 4.0:
                    print(f"[runtime] TIMEOUT catch -> IDLE")
                    hide_yellow_ball()
                    sm_state = STATE_IDLE
                # Si la bola tocó piso sin grab
                elif ball_pos[2] < 0.05:
                    print(f"[runtime] BOLA AL PISO sin grab -> IDLE")
                    hide_yellow_ball()
                    sm_state = STATE_IDLE
            elif sm_state == STATE_HELD:
                # Auto-release tras THROW_HELD_MAX_S. SIN boost — la velocidad
                # sale 100% de la policy. Si la policy no swingueo, la bola cae cerca,
                # y eso es un problema de TRAINING, no del runtime.
                if arm_env.t_grab is not None and (cycle_dt - arm_env.t_grab) > THROW_HELD_MAX_S:
                    arm_env.state = "THROWN"
                    ee_v = arm_env._ee_linvel_world().copy()
                    arm_env._release_ee_velocity = ee_v
                    arm_env.t_release = cycle_dt
                    # La bola hereda la velocidad del EE (ya seteada via manual override cada step
                    # en HELD). NO la modificamos aca — pure policy output.
                    sm_state = STATE_THROWN
                    follow_start_t = sim_t_total
                    policy_forward_v = -float(ee_v[1])
                    print(f"[runtime] RELEASE @ t={cycle_dt:.2f}s -> THROWN "
                          f"(v_fwd={policy_forward_v:.2f} m/s, v_up={ee_v[2]:.2f} m/s, "
                          f"|v|={np.linalg.norm(ee_v):.2f} m/s)")
            elif sm_state == STATE_THROWN:
                # Activar fase FOLLOW (head sigue bola por ~3s)
                if follow_start_t is not None and (sim_t_total - follow_start_t) > 0.2:
                    sm_state = STATE_FOLLOW
                    print(f"[runtime] -> FOLLOW")
            elif sm_state == STATE_FOLLOW:
                # 3s de follow, despues volver a IDLE
                if follow_start_t is not None and (sim_t_total - follow_start_t) > FOLLOW_DURATION_S:
                    print(f"[runtime] FOLLOW done -> IDLE (head vuelve a red ball)")
                    hide_yellow_ball()
                    sm_state = STATE_IDLE
                    target_world_prev = red_target_world.copy()  # reset para no romper tvel
                # O si la bola se va muy lejos
                elif ball_pos[2] < 0.02 or np.linalg.norm(ball_pos) > 5.0:
                    print(f"[runtime] bola fuera de rango -> IDLE")
                    hide_yellow_ball()
                    sm_state = STATE_IDLE
                    target_world_prev = red_target_world.copy()
            elif sm_state == STATE_WAVING:
                # Wave dura DURATION_S (6s en v2) — el WaveAnimation se apaga solo.
                if wave_current is not None and not wave_current.active:
                    print(f"[runtime] WAVING done -> IDLE")
                    wave_current = None
                    wave_start_t = None
                    sm_state = STATE_IDLE

            target_world_prev = head_target.copy()

            # --- 10. Render frame para stream MJPEG ---
            frame_counter += 1
            if frame_counter >= frame_every_n:
                frame_counter = 0
                try:
                    stream_renderer.update_scene(data, camera=stream_cam)
                    rgb = stream_renderer.render()
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG", quality=STREAM_QUALITY)
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    frame_q.put_nowait(buf.getvalue())
                except Exception as e:
                    pass

            # --- 11. Telemetria al cliente (10 Hz) ---
            if sim_t_total - last_telemetry_t >= 0.1:
                last_telemetry_t = sim_t_total
                try:
                    telemetry_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    telemetry_q.put_nowait({
                        "state": sm_state,
                        "ball_z": float(ball_pos[2]),
                        "fps": 50.0,
                    })
                except queue.Full:
                    pass

            # Viewer sync
            if viewer_ctx is not None:
                viewer_ctx.sync()

            # Cadencia 50Hz
            elapsed = time.perf_counter() - loop_start
            sleep_for = sim_dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

            # Log periodico
            now = time.perf_counter()
            if now - last_log_t >= 5.0:
                last_log_t = now
                print(f"[runtime] state={sm_state} sim_t={sim_t_total:.1f}s ball_z={ball_pos[2]*100:.1f}cm")
    except KeyboardInterrupt:
        print("\n[runtime] CTRL+C, saliendo")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()


if __name__ == "__main__":
    main()
