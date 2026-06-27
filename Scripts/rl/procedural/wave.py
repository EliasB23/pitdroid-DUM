"""Saludo procedural del Pit Droid (sin RL).

Coreografía especificada por el usuario:
    1. Retrae el forearm (codo flexionado hacia el cuerpo).
    2. Gira la muñeca medio giro (~90°).
    3. Empieza a volver, abriendo y cerrando los dedos 2 veces.
    4. Vuelve a pose de reposo.

Duración total ≈ 4 s. Es ADITIVO: si el saludo está activo, escribe los joints
del brazo elegido (izq o der); el resto de capas del Animation Engine siguen
controlando sus joints sin conflicto.

Uso:
    wave = WaveAnimation(side="left", model=model)
    wave.trigger(sim_time)
    ...
    ctrl_dict = wave.get_ctrl(sim_time)   # {actuator_name: ctrl_value} o {}
"""

from __future__ import annotations
import math


# Coreografía v4 — SECUENCIAL (no se solapan las fases):
#   1. SUBIR (lento)        la mano sube hasta arriba
#   2. ARRIBA (saludo)      al tope: gira la muñeca + abre/cierra los dedos 2 veces
#   3. BAJAR                recien ahi vuelve a reposo
DURATION_S = 7.5

# Fase 1: SUBIR la mano (retraer forearm). Lento.
T_RETRACT_START = 0.0
T_RETRACT_END   = 3.0     # 3s de subida suave (antes 1.5)

# Fase 2: ARRIBA — el saludo ocurre con la mano ya en el tope.
T_TOP_START     = 3.0
T_TOP_END       = 5.5
# Giro de muñeca: solo durante el tramo de arriba.
T_WRIST_START   = 3.1
T_WRIST_END     = 5.4
# Dedos abriendo/cerrando (2 ciclos): solo durante el tramo de arriba.
T_FINGERS_START = 3.2
T_FINGERS_END   = 5.3
FINGER_CYCLES   = 2

# Fase 3: BAJAR — recien despues del saludo.
T_RETURN_START  = 5.5
T_RETURN_END    = 7.5

T_SETTLE_END    = DURATION_S

# Amplitudes (porcentaje del rango del actuador).
# v4: la mano LLEGA HASTA ARRIBA (forearm alto). El giro de muñeca y los dedos
# son el gesto del saludo. Hombro casi quieto.
FOREARM_RETRACT_AMOUNT = 0.92   # sube alto (antes 0.45) — "llegar hasta arriba"
WRIST_HALF_TURN_AMOUNT = 1.00   # la vuelta de muñeca, gesto del saludo
LEVER_OPEN_AMOUNT      = 1.00   # dedos abriendo/cerrando
SHOULDER_LIFT_AMOUNT   = 0.12   # hombro casi quieto


# Map de nombres de actuadores por lado
ACTUATORS = {
    "left": {
        "shoulder": "act_LeftShoulderArm",
        "forearm":  "act_LeftForearm",
        "wrist":    "act_LeftWrist",
        "lever":    "act_LeftLever_Slider",
    },
    "right": {
        "shoulder": "act_RightShoulderArm",
        "forearm":  "act_RightForearm",
        "wrist":    "act_RightWrist",
        "lever":    "act_RightLever_Slider",
    },
}


def _smoothstep(x: float) -> float:
    """Curva 3x^2 - 2x^3 para transiciones suaves entre 0 y 1."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _ramp(t: float, t_start: float, t_end: float) -> float:
    """Devuelve 0 antes de t_start, 1 después de t_end, smoothstep en medio."""
    if t_end <= t_start:
        return 0.0
    return _smoothstep((t - t_start) / (t_end - t_start))


def _phase_in_window(t: float, t_start: float, t_end: float) -> float | None:
    """Fase 0..1 si t está en la ventana; None fuera."""
    if t < t_start or t > t_end:
        return None
    if t_end <= t_start:
        return None
    return (t - t_start) / (t_end - t_start)


class WaveAnimation:
    """Saludo procedural. Stateful para soportar 'trigger' y consultas por step.

    Resuelve ctrl_lo y ctrl_hi de cada actuador al construirse (necesita el
    `mujoco.MjModel`). En cada step devuelve un dict con los valores absolutos
    a comandar para los 4 actuadores del brazo elegido.
    """

    def __init__(self, side: str, model):
        if side not in ("left", "right"):
            raise ValueError("side debe ser 'left' o 'right'")
        self.side = side
        self.act_names = ACTUATORS[side]
        self._model = model

        # Resolver ids y rangos de los actuadores
        import mujoco
        self._ids = {}
        self._ranges = {}
        for role, name in self.act_names.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"actuator {name!r} no existe en el modelo")
            self._ids[role] = aid
            lo, hi = float(model.actuator_ctrlrange[aid, 0]), float(model.actuator_ctrlrange[aid, 1])
            self._ranges[role] = (lo, hi)

        # Estado de la animación
        self._start_time: float | None = None
        self._active = False
        self._orig_gains = {}   # snapshot de gains/forcerange durante el boost

    # -------------- Boost de actuadores (solo durante el saludo) --------------
    # La muñeca (MG90S, forcerange ±0.21) y el lever (leva con inercia reflejada
    # ×34.9) son fisicamente MUY lentos: con su fuerza nominal los dedos se mueven
    # ~0.6 mm/s y no alcanzan a abrir/cerrar de forma visible. Durante el saludo
    # (animacion scripted, NO el agarre) subimos temporalmente su autoridad para
    # que el gesto se vea, y la restauramos al terminar. El agarre nunca usa esto
    # (saludo y agarre son estados mutuamente excluyentes).
    _BOOST = {"wrist": (2500.0, 2.5), "lever": (5000.0, 10.0)}  # (kp, forcerange)

    def _apply_boost(self):
        m = self._model
        self._orig_gains = {}
        for role, (kp, fr) in self._BOOST.items():
            aid = self._ids[role]
            self._orig_gains[aid] = (
                float(m.actuator_gainprm[aid][0]),
                float(m.actuator_biasprm[aid][1]),
                m.actuator_forcerange[aid].copy(),
            )
            m.actuator_gainprm[aid][0] = kp
            m.actuator_biasprm[aid][1] = -kp
            m.actuator_forcerange[aid][:] = (-fr, fr)

    def _restore_boost(self):
        m = self._model
        for aid, (g, b, fr) in self._orig_gains.items():
            m.actuator_gainprm[aid][0] = g
            m.actuator_biasprm[aid][1] = b
            m.actuator_forcerange[aid][:] = fr
        self._orig_gains = {}

    # -------------- API publica --------------

    def trigger(self, sim_time: float) -> None:
        """Dispara la animación. Si ya estaba activa, reinicia desde 0."""
        self._start_time = sim_time
        self._active = True
        self._apply_boost()

    @property
    def active(self) -> bool:
        return self._active

    def cancel(self) -> None:
        """Aborta el saludo y restaura la fuerza nominal de los actuadores.
        Lo usa el reset suave del motor de animacion."""
        self._active = False
        self._start_time = None
        if self._orig_gains:
            self._restore_boost()

    def get_ctrl(self, sim_time: float) -> dict[str, float]:
        """Devuelve dict {actuator_name: ctrl_value} para los 4 joints del brazo
        si la animación está activa. Si no, dict vacío.

        El llamador (Animation Engine) integra estos valores en el vector de
        control global, sobreescribiendo solo estos actuadores.
        """
        if not self._active or self._start_time is None:
            return {}
        t = sim_time - self._start_time
        if t > DURATION_S:
            self._active = False
            self._restore_boost()   # devolver los actuadores a su fuerza nominal
            return {}

        # --- forearm: retract con smoothstep, vuelta con smoothstep ---
        # retract: 0 → FOREARM_RETRACT_AMOUNT entre T_RETRACT_START..T_RETRACT_END
        # vuelta: FOREARM_RETRACT_AMOUNT → 0 entre T_RETURN_START..T_RETURN_END
        retract_amount = _ramp(t, T_RETRACT_START, T_RETRACT_END) * FOREARM_RETRACT_AMOUNT
        return_amount  = _ramp(t, T_RETURN_START, T_RETURN_END)  * FOREARM_RETRACT_AMOUNT
        forearm_pct = retract_amount - return_amount  # neto

        # --- wrist: half turn ida y vuelta (sin entre T_WRIST_START..T_WRIST_END) ---
        wrist_phase = _phase_in_window(t, T_WRIST_START, T_WRIST_END)
        if wrist_phase is None:
            wrist_pct = 0.0
        else:
            wrist_pct = WRIST_HALF_TURN_AMOUNT * math.sin(math.pi * wrist_phase)

        # --- lever: dos ciclos completos durante la vuelta ---
        fingers_phase = _phase_in_window(t, T_FINGERS_START, T_FINGERS_END)
        if fingers_phase is None:
            lever_pct = 0.0
        else:
            # 0 → 1 → 0 → 1 → 0 en N ciclos
            lever_pct = LEVER_OPEN_AMOUNT * 0.5 * (
                1.0 - math.cos(2.0 * FINGER_CYCLES * math.pi * fingers_phase)
            )

        # --- shoulder: offset MINIMO para que el codo "asome" sin pegarse al cuerpo ---
        # v3: amplitud reducida (SHOULDER_LIFT_AMOUNT) para que casi no tire el hombro atras.
        shoulder_pct = SHOULDER_LIFT_AMOUNT * (
            _ramp(t, T_RETRACT_START, T_RETRACT_END) - _ramp(t, T_RETURN_START, T_RETURN_END)
        )

        ctrl = {
            self.act_names["shoulder"]: self._pct_to_ctrl("shoulder", shoulder_pct),
            self.act_names["forearm"]:  self._pct_to_ctrl("forearm",  forearm_pct),
            self.act_names["wrist"]:    self._pct_to_ctrl("wrist",    wrist_pct),
            self.act_names["lever"]:    self._pct_to_ctrl("lever",    lever_pct),
        }
        return ctrl

    # -------------- helpers --------------

    def _pct_to_ctrl(self, role: str, pct: float) -> float:
        """Mapea pct ∈ [-1, 1] al ctrlrange del actuador.

        - pct = 0  → midpoint (o 0 si el ctrlrange incluye 0)
        - pct = +1 → ctrl_hi
        - pct = -1 → ctrl_lo
        Para el lever (rango 0..max), pct positivo abre (pct=1 → ctrl_hi),
        pct negativo se clipea a 0.
        """
        lo, hi = self._ranges[role]
        if lo >= 0.0:
            # rango [0, hi] (caso lever): solo pct positivo
            return max(0.0, min(hi, pct * hi))
        # rango bilateral [lo, hi]: pct=+1 → hi, pct=-1 → lo
        if pct >= 0.0:
            return min(hi, pct * hi)
        return max(lo, pct * abs(lo))
