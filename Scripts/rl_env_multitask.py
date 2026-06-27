"""DUMMultitaskEnv: env multi-task v14 — head tracking + brazo izquierdo saludando + pinza grip.

Extiende la misma logica que rl_env.DUMHeadTrackingEnv pero:
- Action space: Box(7,)  -> [neck, headbase, headrot, lsh, lfa, lwr, lls]
- Observation space: Box(21,) = 18 obs originales + 3 flags binarios (track, wave, grip).
- Reward: combinacion ponderada de r_track + r_wave + r_grip, activada por flags por episodio.

NO modifica rl_env.py — corre en paralelo como una variante. El XML, los ids del head, target
y lente se resuelven igual que en v13.
"""

from pathlib import Path
import numpy as np
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


# --- Path al modelo MJCF (mismo que v13) ---
XML_PATH = str(Path(__file__).resolve().parents[1] / "Cuerpo" / "DUM4.xml")

# --- Pesos / params de reward originales del head tracking (copiados de rl_env.py) ---
DEFAULT_REWARD_WEIGHTS = dict(
    w_track=1.0,
    sigma=0.4,
    w_smooth=0.15,
    w_effort=0.0075,
    w_alive=0.0,
    w_jitter=0.01,
    w_focus_event=0.0,
    w_focus_hold=0.03,
    w_post_focus_mult=1.5,
)

# --- Hook de enfoque (animacion del lente) ---
FOCUS_THRESHOLD = 0.07
FOCUS_SETTLING_STEPS = 10
FOCUS_ANIM_STEPS = 50
FOCUS_N_CYCLES_RANGE = (1, 4)
FOCUS_AMPLITUDE_RANGE = (0.5, 1.0)
FOCUS_RESTING_RANGE = (0.0, 0.15)

# --- Tarea de tracking ---
TARGET_CONE_AZIMUTH = np.deg2rad(75)
TARGET_CONE_ELEVATION = np.deg2rad(45)
TARGET_DISTANCE_MIN = 0.3
TARGET_DISTANCE_MAX = 1.0
EPISODE_MAX_STEPS = 500

TARGET_MODES = ["static", "linear", "circular"]
TARGET_LINEAR_PERIOD_RANGE = (4.0, 8.0)
TARGET_CIRCULAR_RADIUS_RANGE = (0.05, 0.15)
TARGET_CIRCULAR_OMEGA_RANGE = (0.5, 1.5)

# --- Identificadores del XML (head + lente + arm izquierdo + leva izquierda) ---
HEAD_ACTUATORS = ["act_Neck", "act_HeadBase", "act_HeadRot"]
HEAD_JOINTS = ["Neck_joint", "HeadBase_joint", "HeadRotation_joint"]
LENS_ACTUATOR = "act_LenteExt"
LENS_JOINT = "LenteExt_joint"
HEAD_BODY = "LenteExt_link"

LEFT_ARM_ACTUATORS = ["act_LeftShoulderArm", "act_LeftForearm", "act_LeftWrist"]
LEFT_ARM_JOINTS = ["LeftShoulderArm_joint", "LeftForearm_joint", "LeftWrist_joint"]
LEFT_LEVER_ACTUATOR = "act_LeftLever_Slider"
LEFT_LEVER_JOINT = "LeftLever_Slider"

HEAD_FORWARD_LOCAL = np.array([0.0, -1.0, 0.0])

# --- Multi-task params ---
# Wave: saludo periodico del brazo izquierdo (shoulder + forearm en cuadratura)
WAVE_FREQ_HZ = 0.5
WAVE_AMP_SHOULDER = 0.5
WAVE_AMP_FOREARM = 0.4
WAVE_PHASE = np.pi / 4

# Grip: alterna apertura/cierre de la pinza (slider de la leva izquierda)
GRIP_TARGETS = (0.0, 0.02)
GRIP_SWITCH_RANGE = (2.0, 3.0)  # segundos entre switches

# Pose de "reposo" del brazo cuando flag_wave=0 (shoulder, forearm, wrist) en radianes.
HOME_POSE_ARM = np.array([0.0, 0.0, 0.0])

# Probabilidad de cada flag activado independientemente (se fuerza al menos uno).
FLAG_PROB = 0.6

# Pesos relativos por subtask
W_TRACK = 1.0
W_WAVE = 0.6
W_GRIP = 0.4


class DUMMultitaskEnv(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(self, reward_weights=None, render_mode=None, **kwargs):
        self._reward_w = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}

        # obs = qpos_head(3) + qvel_head(3) + qpos_hb(1) + qvel_hb(1)
        #     + tdir(3) + tdist(1) + tvel(3) + prev_action(3=head!) ... NO
        # Importante: prev_action ahora es de 7 dims (la action_space cambio).
        # Pero para reusar la base v13 dejo el bloque de obs original con prev_action[:3]
        # (la parte de head). En su lugar, sumo los 3 flags al final.
        # obs total: 18 (igual que v13) + 3 (flags) = 21.
        observation_space = Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            model_path=XML_PATH,
            frame_skip=4,
            observation_space=observation_space,
            render_mode=render_mode,
            **kwargs,
        )

        self._resolve_ids()

        # Override del action_space (v13 era 3, ahora 7).
        self.action_space = Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Indice de actuadores que la policy controla (orden = action vector).
        # [neck, headbase, headrot, lsh, lfa, lwr, lls]
        self._policy_actuators = np.array(
            self._head_act_ids + self._left_arm_act_ids + [self._left_lever_act_id],
            dtype=np.int32,
        )
        # Ctrl ranges por actuador (para mapear [-1,1] -> ctrlrange).
        self._policy_ctrl_lo = np.array(
            [self.model.actuator_ctrlrange[a, 0] for a in self._policy_actuators]
        )
        self._policy_ctrl_hi = np.array(
            [self.model.actuator_ctrlrange[a, 1] for a in self._policy_actuators]
        )

        # Estado interno
        self._prev_action = np.zeros(7, dtype=np.float32)
        self._step_count = 0
        self._focus_triggered = False
        self._focus_anim_step = -1
        self._focus_hold_streak = 0
        self._target_world = np.zeros(3)
        self._target_world_prev = np.zeros(3)
        self._focus_n_cycles = 2
        self._focus_amplitude = 1.0
        self._focus_resting = 0.0
        self._target_mode = "static"
        self._target_a = np.zeros(3)
        self._target_b = np.zeros(3)
        self._target_period = 5.0
        self._target_radius = 0.1
        self._target_omega = 1.0
        self._target_phase = 0.0
        self._target_center_for_orbit = np.zeros(3)
        self._head_pos_init = np.zeros(3)

        # Estado de multi-task
        self._flags = np.zeros(3, dtype=np.float32)  # [flag_track, flag_wave, flag_grip]
        self._wave_t0 = 0.0
        self._grip_target = float(GRIP_TARGETS[0])
        self._grip_next_switch = float(GRIP_SWITCH_RANGE[0])

    # -------------------------------------------------------------------------
    # Resolucion de ids
    # -------------------------------------------------------------------------
    def _resolve_ids(self):
        self._head_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in HEAD_ACTUATORS
        ]
        self._head_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in HEAD_JOINTS
        ]
        self._head_qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self._head_joint_ids]
        )
        self._head_dof_adr = np.array(
            [self.model.jnt_dofadr[j] for j in self._head_joint_ids]
        )

        self._lens_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LENS_ACTUATOR
        )
        self._lens_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, LENS_JOINT
        )
        self._lens_max = float(self.model.jnt_range[self._lens_jnt_id, 1])

        hb_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "HipBody_joint"
        )
        self._hipbody_qpos_adr = int(self.model.jnt_qposadr[hb_jid])
        self._hipbody_dof_adr = int(self.model.jnt_dofadr[hb_jid])

        self._lens_center_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "lens_center"
        )

        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._target_mocap_idx = int(self.model.body_mocapid[target_body_id])
        self._head_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY
        )

        # Brazo izquierdo
        self._left_arm_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in LEFT_ARM_ACTUATORS
        ]
        self._left_arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in LEFT_ARM_JOINTS
        ]
        self._left_arm_qpos_adr = np.array(
            [self.model.jnt_qposadr[j] for j in self._left_arm_joint_ids]
        )
        # Adresses individuales para legibilidad en reward
        self._adr_lsh = int(self._left_arm_qpos_adr[0])
        self._adr_lfa = int(self._left_arm_qpos_adr[1])
        self._adr_lwr = int(self._left_arm_qpos_adr[2])

        # Leva izquierda (pinza)
        self._left_lever_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_LEVER_ACTUATOR
        )
        self._left_lever_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, LEFT_LEVER_JOINT
        )
        self._adr_lls = int(self.model.jnt_qposadr[self._left_lever_jnt_id])

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    def reset_model(self):
        self.set_state(self.init_qpos.copy(), self.init_qvel.copy())
        mujoco.mj_forward(self.model, self.data)

        head_pos = self.data.xpos[self._head_body_id].copy()
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3).copy()
        forward_world_init = head_mat @ HEAD_FORWARD_LOCAL
        self._head_pos_init = head_pos.copy()
        self._forward_world_init = forward_world_init.copy()

        mode_idx = int(self.np_random.integers(0, len(TARGET_MODES)))
        self._target_mode = TARGET_MODES[mode_idx]

        self._target_a = self._sample_point_in_cone()
        if self._target_mode == "linear":
            self._target_b = self._sample_point_in_cone()
            self._target_period = float(
                self.np_random.uniform(*TARGET_LINEAR_PERIOD_RANGE)
            )
        elif self._target_mode == "circular":
            self._target_radius = float(
                self.np_random.uniform(*TARGET_CIRCULAR_RADIUS_RANGE)
            )
            self._target_omega = float(
                self.np_random.uniform(*TARGET_CIRCULAR_OMEGA_RANGE)
            )
            self._target_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))
            self._target_center_for_orbit = self._target_a.copy()

        self._update_target_position(sim_time=0.0)
        self.data.mocap_pos[self._target_mocap_idx] = self._target_world

        self.data.ctrl[:] = 0.0
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = np.zeros(7, dtype=np.float32)
        self._step_count = 0
        self._focus_triggered = False
        self._focus_anim_step = -1
        self._focus_hold_streak = 0
        self._target_world_prev = self._target_world.copy()

        self._focus_n_cycles = int(self.np_random.integers(*FOCUS_N_CYCLES_RANGE))
        self._focus_amplitude = float(
            self.np_random.uniform(*FOCUS_AMPLITUDE_RANGE)
        )
        self._focus_resting = (
            float(self.np_random.uniform(*FOCUS_RESTING_RANGE)) * self._lens_max
        )

        # Multi-task: muestrear flags (al menos uno activo).
        flags = (self.np_random.random(size=3) < FLAG_PROB).astype(np.float32)
        if flags.sum() == 0:
            flags[int(self.np_random.integers(0, 3))] = 1.0
        self._flags = flags

        # Reset estado wave/grip
        self._wave_t0 = 0.0
        self._grip_target = float(self.np_random.choice(GRIP_TARGETS))
        self._grip_next_switch = float(
            self.np_random.uniform(*GRIP_SWITCH_RANGE)
        )

        return self._get_obs()

    def _sample_point_in_cone(self):
        az = self.np_random.uniform(-TARGET_CONE_AZIMUTH, TARGET_CONE_AZIMUTH)
        el = self.np_random.uniform(-TARGET_CONE_ELEVATION, TARGET_CONE_ELEVATION)
        dist = self.np_random.uniform(TARGET_DISTANCE_MIN, TARGET_DISTANCE_MAX)
        target_dir = self._rot_z(az) @ self._rot_x(el) @ self._forward_world_init
        target_dir /= np.linalg.norm(target_dir) + 1e-9
        return self._head_pos_init + dist * target_dir

    def _update_target_position(self, sim_time):
        if self._target_mode == "static":
            self._target_world = self._target_a
        elif self._target_mode == "linear":
            phase = 0.5 * (
                1.0 - np.cos(2.0 * np.pi * sim_time / self._target_period)
            )
            self._target_world = self._target_a + phase * (
                self._target_b - self._target_a
            )
        elif self._target_mode == "circular":
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

    # -------------------------------------------------------------------------
    # Geometria de la mirada
    # -------------------------------------------------------------------------
    def _head_forward_world(self):
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        return head_mat @ HEAD_FORWARD_LOCAL

    def _lens_origin_world(self):
        return self.data.site_xpos[self._lens_center_site_id]

    def _target_local(self):
        origin = self._lens_origin_world()
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        return head_mat.T @ (self._target_world - origin)

    def _target_vel_local(self):
        dt = self.frame_skip * float(self.model.opt.timestep)
        head_mat = self.data.xmat[self._head_body_id].reshape(3, 3)
        vel_world = (self._target_world - self._target_world_prev) / dt
        return head_mat.T @ vel_world

    def _angle_to_target(self):
        forward = self._head_forward_world()
        delta = self._target_world - self._lens_origin_world()
        delta /= np.linalg.norm(delta) + 1e-9
        return float(np.arccos(np.clip(np.dot(forward, delta), -1.0, 1.0)))

    # -------------------------------------------------------------------------
    # Observacion
    # -------------------------------------------------------------------------
    def _get_obs(self):
        qpos_head = self.data.qpos[self._head_qpos_adr]
        qvel_head = self.data.qvel[self._head_dof_adr]
        qpos_hb = float(self.data.qpos[self._hipbody_qpos_adr])
        qvel_hb = float(self.data.qvel[self._hipbody_dof_adr])
        tloc = self._target_local()
        tdist = float(np.linalg.norm(tloc))
        tdir = tloc / (tdist + 1e-9)
        tvel = self._target_vel_local()
        # prev_action: solo los 3 head para mantener bloque de 18 igual a v13.
        # Esto preserva la posibilidad de warm-start desde v13.
        prev_head = self._prev_action[:3].astype(np.float64)
        return np.concatenate(
            [
                qpos_head,
                qvel_head,
                [qpos_hb, qvel_hb],
                tdir,
                [tdist],
                tvel,
                prev_head,
                self._flags.astype(np.float64),
            ]
        )

    # -------------------------------------------------------------------------
    # Mapeo de accion a ctrl
    # -------------------------------------------------------------------------
    def _action_to_full_ctrl(self, action):
        a = np.clip(action, -1.0, 1.0).astype(np.float64)
        # Map [-1,1] -> [ctrl_lo, ctrl_hi] por actuador.
        scaled = (a + 1.0) * 0.5 * (
            self._policy_ctrl_hi - self._policy_ctrl_lo
        ) + self._policy_ctrl_lo
        full_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        for i, aid in enumerate(self._policy_actuators):
            full_ctrl[aid] = scaled[i]
        return full_ctrl

    def _lens_focus_ctrl(self):
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

    # -------------------------------------------------------------------------
    # step
    # -------------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(
                f"Action shape esperada (7,), recibida {action.shape}"
            )

        self._target_world_prev = self._target_world.copy()
        sim_time = self._step_count * self.frame_skip * float(
            self.model.opt.timestep
        )
        self._update_target_position(sim_time=sim_time)
        self.data.mocap_pos[self._target_mocap_idx] = self._target_world

        theta_pre = self._angle_to_target()
        focus_just_triggered = False
        if (
            not self._focus_triggered
            and self._step_count >= FOCUS_SETTLING_STEPS
            and theta_pre < FOCUS_THRESHOLD
        ):
            self._focus_triggered = True
            self._focus_anim_step = 0
            focus_just_triggered = True

        full_ctrl = self._action_to_full_ctrl(action)
        full_ctrl[self._lens_act_id] = self._lens_focus_ctrl()

        self.do_simulation(full_ctrl, self.frame_skip)

        if self._focus_anim_step >= 0:
            self._focus_anim_step += 1
            if self._focus_anim_step >= FOCUS_ANIM_STEPS:
                self._focus_anim_step = -1

        # ----------------- REWARD -----------------
        theta_post = self._angle_to_target()
        w = self._reward_w

        in_target_now = theta_post < FOCUS_THRESHOLD
        if in_target_now:
            self._focus_hold_streak += 1
        else:
            self._focus_hold_streak = 0

        focus_mult = w["w_post_focus_mult"] if self._focus_triggered else 1.0

        # Componente shaping del head tracking (smooth, effort, jitter, hold).
        # Estos terminos siguen sumando independientemente del flag de track
        # porque son penalizaciones de estilo que siempre queremos.
        head_action = action[:3]
        head_prev = self._prev_action[:3]
        r_smooth = -w["w_smooth"] * focus_mult * float(
            np.sum((head_action - head_prev) ** 2)
        )
        r_effort = -w["w_effort"] * float(np.sum(head_action ** 2))
        qvel_head = self.data.qvel[self._head_dof_adr]
        r_jitter = -w["w_jitter"] * focus_mult * float(np.sum(qvel_head ** 2))
        r_alive = w["w_alive"]
        r_focus_event = (
            w["w_focus_event"] if focus_just_triggered else 0.0
        )
        r_focus_hold = (
            w["w_focus_hold"] * np.sqrt(self._focus_hold_streak)
            if in_target_now
            else 0.0
        )

        # --- r_track ---
        r_track_raw = float(
            np.exp(-(theta_post ** 2) / (w["sigma"] ** 2))
        )

        # --- r_wave ---
        # Si flag_wave activo: brazo debe seguir la referencia senoidal.
        # Si no: brazo debe mantener HOME_POSE_ARM (quieto).
        t_now = sim_time  # tiempo simulado al inicio del step
        theta_sh_ref = WAVE_AMP_SHOULDER * np.sin(2.0 * np.pi * WAVE_FREQ_HZ * t_now)
        theta_fa_ref = WAVE_AMP_FOREARM * np.sin(
            2.0 * np.pi * WAVE_FREQ_HZ * t_now + WAVE_PHASE
        )
        wave_ref = np.array([theta_sh_ref, theta_fa_ref, 0.0])
        arm_qpos = np.array(
            [
                self.data.qpos[self._adr_lsh],
                self.data.qpos[self._adr_lfa],
                self.data.qpos[self._adr_lwr],
            ]
        )
        if self._flags[1] > 0.5:
            arm_err = arm_qpos - wave_ref
        else:
            arm_err = arm_qpos - HOME_POSE_ARM
        r_wave_raw = float(np.exp(-5.0 * float(np.sum(arm_err ** 2))))

        # --- r_grip ---
        # Switch del target cuando supera self._grip_next_switch (en tiempo sim).
        if t_now >= self._grip_next_switch:
            self._grip_target = (
                GRIP_TARGETS[1]
                if self._grip_target == GRIP_TARGETS[0]
                else GRIP_TARGETS[0]
            )
            self._grip_next_switch = t_now + float(
                self.np_random.uniform(*GRIP_SWITCH_RANGE)
            )
        slider_q = float(self.data.qpos[self._adr_lls])
        r_grip_raw = float(np.exp(-50.0 * (slider_q - self._grip_target) ** 2))

        # Componentes ponderados por flag.
        flag_track = float(self._flags[0])
        flag_wave = float(self._flags[1])
        flag_grip = float(self._flags[2])

        r_track = W_TRACK * flag_track * w["w_track"] * r_track_raw
        r_wave = W_WAVE * flag_wave * r_wave_raw
        r_grip = W_GRIP * flag_grip * r_grip_raw

        reward = (
            r_track
            + r_wave
            + r_grip
            + r_smooth
            + r_effort
            + r_jitter
            + r_alive
            + r_focus_event
            + r_focus_hold
        )

        self._prev_action = action.copy()
        self._step_count += 1

        obs = self._get_obs()
        terminated = False
        truncated = self._step_count >= EPISODE_MAX_STEPS

        currently_in_target = in_target_now and self._focus_triggered
        info = {
            "theta_rad": theta_post,
            "theta_deg": float(np.rad2deg(theta_post)),
            # Componentes brutos (sin pesos ni flags) para diagnostico.
            "r_track_raw": r_track_raw,
            "r_wave_raw": r_wave_raw,
            "r_grip_raw": r_grip_raw,
            # Componentes finales (ya ponderados).
            "r_track": r_track,
            "r_wave": r_wave,
            "r_grip": r_grip,
            "r_smooth": r_smooth,
            "r_effort": r_effort,
            "r_jitter": r_jitter,
            "r_focus_event": r_focus_event,
            "r_focus_hold": r_focus_hold,
            "focus_triggered": currently_in_target,
            "focus_ever_triggered": self._focus_triggered,
            "focus_anim_step": self._focus_anim_step,
            "focus_hold_streak": self._focus_hold_streak,
            "target_mode": self._target_mode,
            # Multi-task diagnostico
            "flag_track": flag_track,
            "flag_wave": flag_wave,
            "flag_grip": flag_grip,
            "grip_target": self._grip_target,
            "slider_q": slider_q,
            "arm_err_norm": float(np.linalg.norm(arm_err)),
        }
        return obs, reward, terminated, truncated, info
