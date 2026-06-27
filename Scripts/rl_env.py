"""DUMHeadTrackingEnv: Gymnasium environment para entrenar una policy
que oriente la cabeza del Pit Droid hacia un target visual.

Hereda de gymnasium.envs.mujoco.MujocoEnv (v5+, bindings nuevos de mujoco).

Action space:
    Box(3,) en [-1, 1] -> mapea a ctrlrange de [act_Neck, act_HeadBase, act_HeadRot].

Observation space:
    Box(13,) = qpos_head(3) + qvel_head(3) + target_dir_local(3) + target_dist(1) + prev_action(3).

Reward (sin LaTeX):
    r = w_track * exp(-theta^2 / sigma^2)
        - w_smooth * ||a_t - a_{t-1}||^2
        - w_effort * ||a_t||^2
        - w_jitter * ||qvel_head||^2
        + w_alive
"""

from pathlib import Path
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


XML_PATH = str(Path(__file__).resolve().parents[1] / "Cuerpo" / "DUM4.xml")

DEFAULT_REWARD_WEIGHTS = dict(
    w_track=1.0,
    sigma=0.4,        # rad — ancho del exponencial de tracking.
                      # Originalmente 0.1, subido tras smoke test:
                      # con 0.1 r_track ≈ 0 a θ>11°, no daba gradient desde poses iniciales.
                      # Con 0.4: r_track(7°)=0.97, r_track(28°)=0.29, r_track(45°)=0.08.
    w_smooth=0.15,    # Subido de 0.05: forzar comandos mas suaves contra chattering.
    w_effort=0.0075,
    w_alive=0.0,
    w_jitter=0.01,    # Subido de 0.002: el v8 segui­a vibrando al asentar.
                      # 0.01 implica ~5x mas penalty por qvel residual.
    w_focus_event=0.0,    # ELIMINADO (era 5.0): el experto lo descarto, mete escalon discreto
                          # en value function y compite con el hold creciente.
    w_focus_hold=0.03,    # Coeficiente para r_hold = w_hold * sqrt(streak_steps) cuando theta<umbral.
                          # sqrt premia mantener prolongado sin explotar (lineal cap saturaria).
    w_post_focus_mult=1.5, # Bajado de 2.0: el reward de hold ya empuja "quietud", no hace falta tanto multiplicador.
    # v14b_head: guidance shaping para forzar uso de HeadRot en azimuth extremo.
    # Penalty proporcional al mismatch entre HeadRot_qpos y el target_az esperado,
    # SOLO cuando |target_az| > umbral (no aplica en frontal).
    w_headrot_guidance=0.0,         # default 0 = sin shaping. v14c uso 2.0 (era 0.5).
    headrot_guidance_az_thr=0.3,    # rad ~17° — debajo de esto, no se aplica el penalty
    # v14c: penalty DIRECTO a |HeadBase_qpos| cuando target_az es grande.
    # v14b fallo porque la policy saturaba HeadBase a ±0.7 rad (limite fisico) para
    # compensar la falta de HeadRot. Este penalty fuerza explicitamente a NO usar
    # HeadBase para azimuth. HeadBase queda libre para elevation moderada.
    w_headbase_tilt_penalty=0.0,    # default 0. v14c usa 1.5
    headbase_tilt_az_thr=0.5,       # rad ~30° — debajo de esto, HeadBase libre
    headbase_tilt_q_thr=0.2,        # rad ~11° — debajo de esto, HeadBase libre
)

# Hook de enfoque animatronico
FOCUS_THRESHOLD = 0.07         # rad ≈ 4°, equivale a r_track > 0.6
FOCUS_SETTLING_STEPS = 10      # ignora los primeros N steps para evitar disparo trivial
FOCUS_ANIM_STEPS = 50          # ~1 segundo a 50 Hz de control
# Rangos aleatorios para variar la anim de enfoque entre episodios:
FOCUS_N_CYCLES_RANGE = (1, 4)         # samplea en [low, high) -> 1, 2 o 3 ciclos
FOCUS_AMPLITUDE_RANGE = (0.5, 1.0)    # fracción del lens_max alcanzada en cada ciclo
FOCUS_RESTING_RANGE = (0.0, 0.15)     # offset final como fracción del lens_max

# Re-trigger del focus (solo si one_shot_focus=False, modo interactivo):
# Si el agente pierde el target sostenidamente, se "rearma" el trigger para que
# vuelva a dispararse cuando recupere el focus.
FOCUS_REARM_LOST_STEPS = 30           # 0.6s a 50Hz fuera de threshold para rearmar
FOCUS_REARM_THRESHOLD_MULT = 3.0      # debe estar a > 3*FOCUS_THRESHOLD para "perdido"

# Tarea
TARGET_CONE_AZIMUTH = np.deg2rad(75)   # Subido de 60° — exigir mas yaw, forzar a usar HeadRot
TARGET_CONE_ELEVATION = np.deg2rad(45) # Subido de 30° — exigir mas pitch
TARGET_DISTANCE_MIN = 0.3
TARGET_DISTANCE_MAX = 1.0
EPISODE_MAX_STEPS = 500        # 10 s a 50 Hz

# Movimiento del target en cada episodio (se elige uno random):
TARGET_MODES = ["static", "linear", "circular"]
# Linear: oscila entre dos puntos del cono A<->B con periodo:
TARGET_LINEAR_PERIOD_RANGE = (4.0, 8.0)   # segundos
# Circular: orbita alrededor de un punto del cono en un plano perpendicular a la mirada:
TARGET_CIRCULAR_RADIUS_RANGE = (0.05, 0.15)  # m
TARGET_CIRCULAR_OMEGA_RANGE = (0.5, 1.5)     # rad/s ~ 30-90°/s

# Identificadores del XML
HEAD_ACTUATORS = ["act_Neck", "act_HeadBase", "act_HeadRot"]
HEAD_JOINTS = ["Neck_joint", "HeadBase_joint", "HeadRotation_joint"]
LENS_ACTUATOR = "act_LenteExt"
LENS_JOINT = "LenteExt_joint"
HEAD_BODY = "LenteExt_link"    # body cuyo frame se usa para "donde mira"

# Eje "forward" de la mirada, en el frame local de HEAD_BODY.
# Esta es una hipotesis inicial — el slide del lente avanza en -Y local,
# por lo tanto el frente del ojo apunta en -Y. Se valida con un test visual.
HEAD_FORWARD_LOCAL = np.array([0.0, -1.0, 0.0])


class DUMHeadTrackingEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, reward_weights=None, render_mode=None,
                 one_shot_focus=True, cone_az_deg=None, cone_el_deg=None, **kwargs):
        """one_shot_focus=True (default, modo training): el focus se dispara una sola
        vez por episodio. one_shot_focus=False (modo interactivo): el trigger se rearma.

        cone_az_deg / cone_el_deg: override del cono de muestreo del target en GRADOS.
        Si None, usa los defaults TARGET_CONE_AZIMUTH/ELEVATION (75°/45°)."""
        self._reward_w = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self._one_shot_focus = one_shot_focus
        # v14b: cono de muestreo del target (instance-level, overridable)
        self._cone_az = np.deg2rad(cone_az_deg) if cone_az_deg is not None else TARGET_CONE_AZIMUTH
        self._cone_el = np.deg2rad(cone_el_deg) if cone_el_deg is not None else TARGET_CONE_ELEVATION

        # Obs: qpos_head(3) + qvel_head(3) + qpos_hipbody(1) + qvel_hipbody(1)
        #    + target_dir_local(3) + target_dist(1) + target_vel_local(3) + prev_action(3) = 18
        # target_vel_local agregada en v13 para que la policy anticipe el movimiento del target.
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path=XML_PATH,
            frame_skip=4,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )

        # Resolver ids una sola vez
        self._head_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in HEAD_ACTUATORS
        ]
        self._head_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in HEAD_JOINTS
        ]
        self._head_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self._head_joint_ids])
        self._head_dof_adr = np.array([self.model.jnt_dofadr[j] for j in self._head_joint_ids])

        self._lens_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LENS_ACTUATOR)
        self._lens_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, LENS_JOINT)
        self._lens_max = float(self.model.jnt_range[self._lens_jnt_id, 1])

        # HipBody_joint para incluir en la observacion: el agente "ve" el torso oscilando
        hb_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "HipBody_joint")
        self._hipbody_qpos_adr = int(self.model.jnt_qposadr[hb_jid])
        self._hipbody_dof_adr = int(self.model.jnt_dofadr[hb_jid])

        # Site lens_center: origen geometrico real del rayo de mirada
        # (resuelve "desfase constante" causado por el offset body.xpos -> centro del lente)
        self._lens_center_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "lens_center"
        )

        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_idx = int(self.model.body_mocapid[target_body_id])
        self._head_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY)

        # Override del action_space (MujocoEnv default lo deriva del XML — 14 dims)
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._head_ctrl_lo = np.array(
            [self.model.actuator_ctrlrange[a, 0] for a in self._head_act_ids]
        )
        self._head_ctrl_hi = np.array(
            [self.model.actuator_ctrlrange[a, 1] for a in self._head_act_ids]
        )

        # Estado interno (se reinicia en reset_model)
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._focus_triggered = False
        self._focus_anim_step = -1
        self._focus_hold_streak = 0   # steps consecutivos con theta < umbral
        self._steps_out_of_focus = 0  # solo se usa cuando one_shot_focus=False (re-arm)
        self._target_world = np.zeros(3)
        self._target_world_prev = np.zeros(3)  # para diferencia finita -> target_vel_local
        # Parametros aleatorios de la anim de enfoque (resamplear en reset)
        self._focus_n_cycles = 2
        self._focus_amplitude = 1.0
        self._focus_resting = 0.0
        # Parametros del movimiento del target (resamplear en reset)
        self._target_mode = "static"
        self._target_a = np.zeros(3)
        self._target_b = np.zeros(3)
        self._target_period = 5.0
        self._target_radius = 0.1
        self._target_omega = 1.0
        self._target_phase = 0.0
        self._target_center_for_orbit = np.zeros(3)
        self._head_pos_init = np.zeros(3)

    # ---------- Reset / sampleo del target ----------

    def reset_model(self):
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        mujoco.mj_forward(self.model, self.data)

        # Origen del target = posicion world del head body en pose neutra
        head_pos = self.data.xpos[self._head_body_id].copy()
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3).copy()
        forward_world_init = head_mat @ HEAD_FORWARD_LOCAL
        self._head_pos_init = head_pos.copy()
        self._forward_world_init = forward_world_init.copy()

        # ---- Sampleo del modo de movimiento ----
        mode_idx = int(self.np_random.integers(0, len(TARGET_MODES)))
        self._target_mode = TARGET_MODES[mode_idx]

        # Sample primer punto (siempre); para "static" es el unico
        self._target_a = self._sample_point_in_cone()

        if self._target_mode == "linear":
            self._target_b = self._sample_point_in_cone()
            self._target_period = float(self.np_random.uniform(*TARGET_LINEAR_PERIOD_RANGE))
        elif self._target_mode == "circular":
            self._target_radius = float(self.np_random.uniform(*TARGET_CIRCULAR_RADIUS_RANGE))
            self._target_omega = float(self.np_random.uniform(*TARGET_CIRCULAR_OMEGA_RANGE))
            self._target_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            self._target_center_for_orbit = self._target_a.copy()

        # Setear target inicial
        self._update_target_position(sim_time=0.0)
        self.data.mocap_pos[self._target_mocap_idx] = self._target_world

        # Settling con ctrl=0 para que la fisica se asiente
        self.data.ctrl[:] = 0.0
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Estado interno
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._focus_triggered = False
        self._focus_anim_step = -1
        self._focus_hold_streak = 0
        self._target_world_prev = self._target_world.copy()

        # Aleatorizar la anim de enfoque para variedad visual
        self._focus_n_cycles = int(self.np_random.integers(*FOCUS_N_CYCLES_RANGE))
        self._focus_amplitude = float(self.np_random.uniform(*FOCUS_AMPLITUDE_RANGE))
        self._focus_resting = float(self.np_random.uniform(*FOCUS_RESTING_RANGE)) * self._lens_max

        return self._get_obs()

    def set_cone_limits(self, az_rad: float, el_rad: float):
        """v14_head: callback puede actualizar los limites del cono per-step.
        Si nunca se llama, se usan los defaults (TARGET_CONE_*)."""
        self._cone_az = float(az_rad)
        self._cone_el = float(el_rad)

    def _sample_point_in_cone(self):
        """Muestrea un punto en el cono frontal a partir de head_pos_init."""
        # v14_head: usar limites instance-level si se setearon, sino los globales
        az_lim = getattr(self, "_cone_az", TARGET_CONE_AZIMUTH)
        el_lim = getattr(self, "_cone_el", TARGET_CONE_ELEVATION)
        az = self.np_random.uniform(-az_lim, az_lim)
        el = self.np_random.uniform(-el_lim, el_lim)
        dist = self.np_random.uniform(TARGET_DISTANCE_MIN, TARGET_DISTANCE_MAX)
        target_dir = self._rot_z(az) @ self._rot_x(el) @ self._forward_world_init
        target_dir /= np.linalg.norm(target_dir) + 1e-9
        return self._head_pos_init + dist * target_dir

    def _update_target_position(self, sim_time):
        """Actualiza self._target_world segun el modo y el tiempo simulado."""
        if self._target_mode == "static":
            self._target_world = self._target_a
        elif self._target_mode == "linear":
            # phase oscila 0..1..0..1.. con periodo T (coseno)
            phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * sim_time / self._target_period))
            self._target_world = self._target_a + phase * (self._target_b - self._target_a)
        elif self._target_mode == "circular":
            # Orbita en un plano perpendicular a (target_a - head_pos)
            radial = self._target_center_for_orbit - self._head_pos_init
            r_norm = np.linalg.norm(radial)
            if r_norm < 1e-6:
                radial = np.array([0.0, -1.0, 0.0])
                r_norm = 1.0
            radial = radial / r_norm
            up = np.array([0.0, 0.0, 1.0])
            u = np.cross(radial, up)
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(radial, np.array([1.0, 0.0, 0.0]))
            u = u / (np.linalg.norm(u) + 1e-9)
            v = np.cross(radial, u)
            v = v / (np.linalg.norm(v) + 1e-9)
            ang = self._target_phase + self._target_omega * sim_time
            offset = self._target_radius * (np.cos(ang) * u + np.sin(ang) * v)
            self._target_world = self._target_center_for_orbit + offset

    @staticmethod
    def _rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def _rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # ---------- Geometria de la mirada ----------

    def _head_forward_world(self):
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        return head_mat @ HEAD_FORWARD_LOCAL

    def _lens_origin_world(self):
        """Origen geometrico real del rayo de mirada (centro visual del lente)."""
        return self.data.site_xpos[self._lens_center_site_id]

    def _target_local(self):
        """Vector cabeza->target en frame local del lente. Origen: centro visual del lente."""
        origin = self._lens_origin_world()
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        return head_mat.T @ (self._target_world - origin)

    def _target_vel_local(self):
        """Velocidad del target en frame local del lente (diferencia finita)."""
        dt = self.frame_skip * float(self.model.opt.timestep)
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        vel_world = (self._target_world - self._target_world_prev) / dt
        return head_mat.T @ vel_world

    def _angle_to_target(self):
        forward = self._head_forward_world()
        delta = self._target_world - self._lens_origin_world()
        delta /= np.linalg.norm(delta) + 1e-9
        return float(np.arccos(np.clip(np.dot(forward, delta), -1.0, 1.0)))

    # ---------- Observacion ----------

    def _get_obs(self):
        qpos_head = self.data.qpos[self._head_qpos_adr]
        qvel_head = self.data.qvel[self._head_dof_adr]
        qpos_hb = float(self.data.qpos[self._hipbody_qpos_adr])
        qvel_hb = float(self.data.qvel[self._hipbody_dof_adr])
        tloc = self._target_local()
        tdist = float(np.linalg.norm(tloc))
        tdir = tloc / (tdist + 1e-9)
        tvel = self._target_vel_local()  # nuevo en v13: anticipacion del movimiento
        return np.concatenate(
            [qpos_head, qvel_head, [qpos_hb, qvel_hb], tdir, [tdist], tvel,
             self._prev_action.astype(np.float64)]
        )

    # ---------- Mapeo de accion a ctrl ----------

    def _action_to_head_ctrl(self, action):
        a = np.clip(action, -1.0, 1.0).astype(np.float64)
        # [-1,1] -> [ctrl_lo, ctrl_hi]
        return (a + 1.0) * 0.5 * (self._head_ctrl_hi - self._head_ctrl_lo) + self._head_ctrl_lo

    def _lens_focus_ctrl(self):
        """Anim de enfoque randomizada por episodio:
        - N ciclos (1, 2 o 3) entre 0 y amplitude*lens_max
        - Tras la anim, queda en resting offset (cerca de 0, leve aleatoriedad)
        """
        if not self._focus_triggered:
            return 0.0
        e = self._focus_anim_step
        if e < 0:
            return 0.0
        if e >= FOCUS_ANIM_STEPS:
            return self._focus_resting
        phase = e / FOCUS_ANIM_STEPS
        peak = self._focus_amplitude * self._lens_max
        return float(peak * 0.5 * (1 - np.cos(2 * self._focus_n_cycles * np.pi * phase)))

    # ---------- step ----------

    def step(self, action):
        # Guardar posicion del target previa para calcular target_vel_local
        self._target_world_prev = self._target_world.copy()

        # Actualizar posicion del target segun modo y tiempo (movimiento del punto rojo)
        sim_time = self._step_count * self.frame_skip * float(self.model.opt.timestep)
        self._update_target_position(sim_time=sim_time)
        self.data.mocap_pos[self._target_mocap_idx] = self._target_world

        # Trigger del hook de enfoque (mide antes del paso)
        theta_pre = self._angle_to_target()
        focus_just_triggered = False

        # Re-arm: si one_shot_focus=False (modo interactivo) y el agente perdio el target
        # sostenidamente, resetear el flag para que pueda volver a dispararse.
        if not self._one_shot_focus and self._focus_triggered:
            if theta_pre > FOCUS_THRESHOLD * FOCUS_REARM_THRESHOLD_MULT:
                self._steps_out_of_focus += 1
                if self._steps_out_of_focus >= FOCUS_REARM_LOST_STEPS:
                    self._focus_triggered = False
                    self._focus_anim_step = -1  # tambien resetear anim por las dudas
                    self._steps_out_of_focus = 0
                    # Resamplear parametros aleatorios de la anim para variedad
                    self._focus_n_cycles = int(self.np_random.integers(*FOCUS_N_CYCLES_RANGE))
                    self._focus_amplitude = float(self.np_random.uniform(*FOCUS_AMPLITUDE_RANGE))
                    self._focus_resting = float(self.np_random.uniform(*FOCUS_RESTING_RANGE)) * self._lens_max
            else:
                self._steps_out_of_focus = 0

        if (
            not self._focus_triggered
            and self._step_count >= FOCUS_SETTLING_STEPS
            and theta_pre < FOCUS_THRESHOLD
        ):
            self._focus_triggered = True
            self._focus_anim_step = 0
            focus_just_triggered = True

        # Construir el vector ctrl completo
        full_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        head_ctrl = self._action_to_head_ctrl(action)
        for i, aid in enumerate(self._head_act_ids):
            full_ctrl[aid] = head_ctrl[i]
        full_ctrl[self._lens_act_id] = self._lens_focus_ctrl()

        # Simular frame_skip pasos
        self.do_simulation(full_ctrl, self.frame_skip)

        # Avanzar animacion de enfoque
        if self._focus_anim_step >= 0:
            self._focus_anim_step += 1
            if self._focus_anim_step >= FOCUS_ANIM_STEPS:
                self._focus_anim_step = -1

        # Reward
        theta_post = self._angle_to_target()
        w = self._reward_w

        # Actualizar streak de hold: incrementa mientras este en target, reset cuando sale.
        in_target_now = theta_post < FOCUS_THRESHOLD
        if in_target_now:
            self._focus_hold_streak += 1
        else:
            self._focus_hold_streak = 0

        # Multiplicador para smooth/jitter cuando ya esta en focus (premia quedarse quieto)
        focus_mult = w["w_post_focus_mult"] if self._focus_triggered else 1.0

        r_track = w["w_track"] * float(np.exp(-(theta_post ** 2) / (w["sigma"] ** 2)))
        r_smooth = -w["w_smooth"] * focus_mult * float(np.sum((action - self._prev_action) ** 2))
        r_effort = -w["w_effort"] * float(np.sum(action ** 2))
        qvel_head = self.data.qvel[self._head_dof_adr]
        r_jitter = -w["w_jitter"] * focus_mult * float(np.sum(qvel_head ** 2))
        r_alive = w["w_alive"]

        # focus_event: w=0 por default (eliminado por el experto). Lo dejo por compat.
        r_focus_event = w["w_focus_event"] if focus_just_triggered else 0.0
        # focus_hold creciente con sqrt(streak): premia mantener prolongado sin explotar.
        r_focus_hold = w["w_focus_hold"] * np.sqrt(self._focus_hold_streak) if in_target_now else 0.0

        # v14b: HeadRot guidance — penalty si HeadRot_qpos no apunta al target_az
        # SOLO cuando el target esta lejos en azimuth (>umbral) y el HeadRot esta
        # lejos del valor esperado. Empuja al policy a usar HeadRot en vez de
        # compensar con HeadBase tilt.
        r_headrot_guidance = 0.0
        r_headbase_tilt_penalty = 0.0
        if w.get("w_headrot_guidance", 0.0) > 0.0 or w.get("w_headbase_tilt_penalty", 0.0) > 0.0:
            # Target azimuth relativo al forward del robot (-Y world)
            tgt_dx = float(self._target_world[0] - self._head_pos_init[0])
            tgt_dy_forward = -float(self._target_world[1] - self._head_pos_init[1])
            target_az = float(np.arctan2(tgt_dx, tgt_dy_forward))
            abs_target_az = abs(target_az)
            # HeadRot guidance
            if w.get("w_headrot_guidance", 0.0) > 0.0:
                az_thr = w.get("headrot_guidance_az_thr", 0.3)
                if abs_target_az > az_thr:
                    head_rot_q = float(self.data.qpos[self._head_qpos_adr[2]])
                    diff = head_rot_q - target_az
                    diff = float(np.arctan2(np.sin(diff), np.cos(diff)))
                    r_headrot_guidance = -w["w_headrot_guidance"] * abs(diff)
            # v14c: HeadBase tilt penalty — penalty directo al uso de HeadBase
            # cuando el target requiere yaw (azimuth grande). HeadBase libre
            # cuando target_az es chico (entonces puede pitch para elevation).
            if w.get("w_headbase_tilt_penalty", 0.0) > 0.0:
                az_thr_b = w.get("headbase_tilt_az_thr", 0.5)
                q_thr = w.get("headbase_tilt_q_thr", 0.2)
                if abs_target_az > az_thr_b:
                    head_base_q = float(self.data.qpos[self._head_qpos_adr[1]])
                    excess = max(0.0, abs(head_base_q) - q_thr)
                    r_headbase_tilt_penalty = -w["w_headbase_tilt_penalty"] * excess

        reward = (r_track + r_smooth + r_effort + r_jitter + r_alive
                  + r_focus_event + r_focus_hold + r_headrot_guidance
                  + r_headbase_tilt_penalty)

        self._prev_action = action.astype(np.float32).copy()
        self._step_count += 1

        obs = self._get_obs()
        terminated = False
        truncated = self._step_count >= EPISODE_MAX_STEPS

        # focus_triggered es el flag interno one-shot (para el efecto del lente).
        # Para el indicador VISUAL queremos algo dinamico: "currently in target".
        currently_in_target = in_target_now and self._focus_triggered
        info = {
            "theta_rad": theta_post,
            "theta_deg": float(np.rad2deg(theta_post)),
            "r_track": r_track,
            "r_smooth": r_smooth,
            "r_effort": r_effort,
            "r_jitter": r_jitter,
            "r_focus_event": r_focus_event,
            "r_focus_hold": r_focus_hold,
            "r_headrot_guidance": r_headrot_guidance,
            "r_headbase_tilt_penalty": r_headbase_tilt_penalty,
            "focus_triggered": currently_in_target,  # estado VISUAL dinamico
            "focus_ever_triggered": self._focus_triggered,  # one-shot interno
            "focus_anim_step": self._focus_anim_step,
            "focus_hold_streak": self._focus_hold_streak,
            "target_mode": self._target_mode,
        }
        return obs, reward, terminated, truncated, info
