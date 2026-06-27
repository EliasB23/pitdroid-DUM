"""Modo interactivo: corre el robot con la policy v13 entrenada,
target controlado en tiempo real desde un browser via WebSocket.

Lanza dos cosas en paralelo:
- Sim loop con MuJoCo viewer (thread principal).
- Servidor FastAPI con WebSocket (thread daemon).

Acceso local:  http://localhost:8000
Acceso LAN:    http://<ip_de_la_pc>:8000  (mismo wifi, abrir puerto 8000 en firewall)

Uso:
    python Scripts/run_interactive.py
    python Scripts/run_interactive.py --policy runs/ppo_dum_v13/final.zip --host 0.0.0.0 --port 8000
"""

import argparse
import io
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
from stable_baselines3 import PPO

from rl_env import DUMHeadTrackingEnv
import web_remote.server as server


class InteractiveEnv(DUMHeadTrackingEnv):
    """Subclase del env de training que toma el target desde una cola
    en vez de muestrearlo segun los modos pre-programados.
    Nunca trunca: la sesion dura indefinidamente."""

    def __init__(self, target_queue: queue.Queue, **kwargs):
        # En modo interactivo el focus debe re-dispararse cada vez que el agente
        # recupera el target (no solo una vez por sesion como en training).
        kwargs.setdefault("one_shot_focus", False)
        super().__init__(**kwargs)
        self._target_queue = target_queue

    def _update_target_position(self, sim_time):
        """Override: target viene de la cola, no de las trayectorias predefinidas."""
        try:
            data = self._target_queue.get_nowait()
            self._target_world = np.array(
                [float(data["x"]), float(data["y"]), float(data["z"])],
                dtype=np.float64,
            )
        except queue.Empty:
            pass  # mantener el target previo

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return obs, r, term, False, info  # nunca truncar en modo interactivo


def run_uvicorn_in_thread(host: str, port: int):
    """Lanza uvicorn en un thread daemon para no bloquear el sim loop."""
    import uvicorn
    config = uvicorn.Config(server.app, host=host, port=port, log_level="warning")
    server_instance = uvicorn.Server(config)
    th = threading.Thread(target=server_instance.run, daemon=True, name="uvicorn")
    th.start()
    return th


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "runs" / "ppo_dum_v13" / "final.zip"),
                   help="Ruta al .zip de la policy entrenada (default: v13).")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="Bind host. 0.0.0.0 = accesible por LAN. 127.0.0.1 = solo local.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--target-smooth", type=float, default=0.3,
                   help="Coef LPF para suavizar target externo (0..1). 1=sin suavizado, 0.3=suave.")
    args = p.parse_args()

    # Colas de comunicacion sim <-> WS
    target_q = queue.Queue(maxsize=1)
    telemetry_q = queue.Queue(maxsize=1)
    frame_q = queue.Queue(maxsize=1)  # JPEG bytes del render del viewer
    server.init_queues(target_q, telemetry_q, frame_q)

    # Lanzar servidor WS
    run_uvicorn_in_thread(args.host, args.port)
    print(f"[server] http://{args.host}:{args.port}  (use http://127.0.0.1:{args.port} desde esta PC)")
    print(f"[server] desde celular en la misma red, usar la IP local del host")

    # Cargar env + policy
    env = InteractiveEnv(target_q)
    obs, _ = env.reset(seed=0)
    print(f"[sim] env OK, obs.shape = {obs.shape}")
    rl_model = PPO.load(args.policy)
    print(f"[sim] policy cargada: {args.policy}")

    # Renderer offscreen para el stream MJPEG a la web.
    # 480x480 / cada 2 steps de control = ~25 fps de video, suficiente.
    stream_renderer = mujoco.Renderer(env.model, height=480, width=480)
    frame_every_n_steps = 2
    frame_step_counter = 0
    print("[sim] renderer offscreen 480x480 listo (MJPEG stream en /stream)")

    # Lanzar viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("[sim] viewer abierto. Cerralo para terminar.")

        target_smooth = float(np.clip(args.target_smooth, 0.0, 1.0))
        sim_dt = float(env.model.opt.timestep) * env.frame_skip  # tiempo entre steps de control (0.02 s)
        last_fps_time = time.perf_counter()
        fps_counter = 0
        measured_fps = 50.0
        telemetry_counter = 0

        while viewer.is_running():
            loop_start = time.perf_counter()

            # Suavizado opcional del target (LPF de primer orden)
            if target_smooth < 1.0 and not target_q.empty():
                # Peek the queue, blend
                try:
                    new_data = target_q.queue[0]  # peek
                    new_target = np.array([new_data["x"], new_data["y"], new_data["z"]])
                    env._target_world = (
                        target_smooth * new_target
                        + (1 - target_smooth) * env._target_world
                    )
                    # Replace in queue with smoothed value to avoid stale jumps
                    try:
                        target_q.get_nowait()
                    except queue.Empty:
                        pass
                    target_q.put_nowait({
                        "x": float(env._target_world[0]),
                        "y": float(env._target_world[1]),
                        "z": float(env._target_world[2]),
                    })
                except (IndexError, queue.Empty, queue.Full):
                    pass

            # Predict y step
            obs = env._get_obs()
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)

            # Telemetria al cliente (10 Hz)
            telemetry_counter += 1
            if telemetry_counter >= 5:
                telemetry_counter = 0
                payload = {
                    "theta_deg": float(info["theta_deg"]),
                    "focus": bool(info["focus_triggered"]),
                    "fps": float(measured_fps),
                }
                try:
                    telemetry_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    telemetry_q.put_nowait(payload)
                except queue.Full:
                    pass

            # Render frame para el stream MJPEG (cada N steps para no matar el FPS)
            frame_step_counter += 1
            if frame_step_counter >= frame_every_n_steps:
                frame_step_counter = 0
                try:
                    stream_renderer.update_scene(env.data)
                    rgb = stream_renderer.render()  # ndarray HxWx3 uint8
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
                    jpeg_bytes = buf.getvalue()
                    # Mantener solo el ultimo frame en la cola
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        frame_q.put_nowait(jpeg_bytes)
                    except queue.Full:
                        pass
                except Exception as e:
                    # No romper el sim loop si falla el render por algun motivo.
                    print(f"[stream] render fallo: {e}")

            # Sync viewer
            viewer.sync()

            # FPS measurement
            fps_counter += 1
            now = time.perf_counter()
            if now - last_fps_time >= 1.0:
                measured_fps = fps_counter / (now - last_fps_time)
                fps_counter = 0
                last_fps_time = now

            # Mantener cadencia de 50 Hz (sim_dt = 20 ms)
            elapsed = time.perf_counter() - loop_start
            sleep_for = sim_dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    print("[sim] viewer cerrado. Saliendo.")


if __name__ == "__main__":
    main()
