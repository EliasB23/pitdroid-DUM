# Reward Shaping definitivo — DUMGrabEnv

> **Generado por agente experto.** Plan ejecutable para entrenar la policy de "agarrar y tirar".

## State machine

```
SPAWNED → FALLING → HELD → THROWN → LANDED_OK  (éxito + reward distancia)
                 ↓               ↓
              FLOOR_FAIL    FLOOR_OK_AFTER_THROW
              (−800)        (+reward por distancia)
                 ↓
              HELD → TIMEOUT_DROP (>3s, penalty exp)
```

**Estados:**
- `FALLING`: cae libre, nunca agarrada.
- `HELD`: equality activa, cronómetro `t_held`.
- `THROWN`: equality desactivada después de `HELD`, vuela libre.

**Terminación:**
- `ball.z < z_floor + 0.05` en cualquier estado.
- `t_episode > 8 s`.
- `t_held > 4 s` (drop forzado anti-cheating).

## Detección de agarre

```python
grabbed = (dist(EE, ball) < 0.06)  and  (lever_qpos > 0.007)  and  (ball.z > 0.10)
```
Al cumplirse por primera vez:
- `model.eq_active[connect_id] = 1`
- `was_held = True`
- `t_grab = t`
- transición a `HELD`

Requiere `<equality type="connect">` precreado en el XML entre `EE_site` y `ball_body`, inicializado con `active="false"`.

## Mecánica del throw

```python
release = (state == HELD)  and  (lever_qpos < 0.003)  and  (lever_qvel < -0.05)
```
Al disparar: `eq_active = 0`. **Patch necesario** (MuJoCo a veces deja velocidad residual cero tras connect): copiar manualmente `qvel_EE` al freejoint de la bola:
```python
data.qvel[ball_adr : ball_adr+3] = ee_linvel
```
Transición a `THROWN`.

## Reward shaping (con constantes)

| Componente | Fórmula | Peso/constantes | State |
|---|---|---|---|
| `r_approach` | −α·dist(EE, ball) − β·‖ball_vel_rel‖ | α=2.0, β=0.1 | FALLING |
| `r_grab_first` | +B_grab (one-shot) | **B_grab = 500** | transición FALLING→HELD |
| `r_grab_height` | +K_h · clip(z_grab, 0.2, 1.2) | K_h = 200 → max ≈ 240 | one-shot al agarrar |
| `r_grab_speed` | +K_s · max(0, 1−(t_grab−t_spawn)/T_max) | K_s=150, T_max=3.0s | one-shot al agarrar |
| `r_hold_dynamics` | piecewise (ver abajo) | — | HELD |
| `r_throw_distance` | +K_d · tanh(d_xy / D_sat) | K_d=400, D_sat=2.0 m | al aterrizar tras THROWN |
| `r_floor_fail` | −P_fail | **P_fail = 800** | terminal si no was_held |
| `r_smooth` | −λ_s · ‖a_t − a_{t−1}‖² | λ_s = 0.05 | siempre |
| `r_effort` | −λ_e · ‖τ‖² | λ_e = 1e-4 | siempre |

### r_hold_dynamics piecewise (por step a 50 Hz)

```python
τ = t − t_grab
if τ ≤ 2.0:   r = +H₀ · (1 − τ/2.0)              # H₀ = 8 → integra ~160 sobre 2s
elif τ ≤ 3.0: r = 0                              # zona neutra
else:         r = −γ · (exp((τ−3.0)/0.5) − 1)    # γ = 4, explota a τ ≈ 4s
```

### Magnitudes verificadas

- **Éxito completo** (grab alto + rápido + throw lejos) ≈ +500+240+150+160+400 = **+1450**.
- **Fail al piso** = **−800**.
- **Hold infinito** = −∞.
- `r_approach` denso acota ruido en ±50 por episodio.

## Distinción "el robot tiró" vs "cayó solo"

Flag `self.was_held`. Al tocar piso:
- `was_held == False` → `r_floor_fail`
- `was_held == True` → `r_throw_distance`

**Edge case `bumped`:** la bola roza el EE sin que se active equality y sale despedida → trataría como FALLING y daría fail. Mitigación: durante FALLING, si `dist(EE,ball) < 0.08` y `ball_vel · normal_EE > 0`, marcar `bumped=True` y aplicar `r_floor_fail · 0.3` (penalty parcial).

## Distancia de tiro

```python
d_xy = ‖ball.xy − base_link.xy‖   # en el frame del touchdown
```

**Saturación con tanh** en `D_sat = 2 m` — tirar a 5m da casi lo mismo que a 3m. Evita exploits de "throw cohete" a costa del agarre. Bonus opcional para récords: `+10 · max(0, d_xy − 2.0)` con cap a +50.

## Curriculum (3 fases)

| Fase | Steps | Setup | Términos activos |
|---|---|---|---|
| **1** | 1.0M | bola **estática** colgada a z=0.8m | approach, grab_first, grab_height, hold_dynamics (solo tramo 0-2s), smooth, effort. **Sin throw, sin floor_fail.** Termina al agarrar o timeout 5s. |
| **2** | 1.0M | caída lenta `g=3.0` | + grab_speed, floor_fail (a 0.3× peso), hold_dynamics completo. Throw opcional con `K_d` reducido a 100. |
| **3** | 1.5M | `g=9.81`, spawn aleatorio z∈[1.2, 1.8] | **Todos los términos a peso pleno. Throw entra acá con `K_d=400`.** |

**Total**: 3.5M steps.

## Cheats anticipados y mitigaciones

1. **Grab+release inmediato para cobrar `r_grab_first` repetido** → flag `_grab_bonus_consumed` por episodio (one-shot).
2. **No cerrar del todo para evitar HELD timer** → threshold `lever > 0.007` estricto. Si dist<0.06 y lever flota en 0.005 por >0.5s con bola cerca, aplicar `r_indecision = −0.5/step`.
3. **Golpear la bola lejos sin agarrar** para fingir throw → cubierto por `bumped` flag (penalty parcial, no computa `r_throw_distance` sin `was_held`).
4. **Spamear `r_approach`** quedándose pegado sin agarrar → `r_approach` se anula al entrar HELD; además decae lineal con dist.
5. **Throw vertical para reagarrar** → `r_grab_first` es one-shot por episodio.

## Estructura del archivo `grab_env.py`

```python
class DUMGrabEnv(MujocoEnv):
    def __init__(self, phase=3):
        self.phase = phase
        self._setup_weights()
        self.connect_eq_id = mj_name2id(model, EQUALITY, "ee_ball_connect")

    def reset(self):
        self.state = "FALLING"
        self.was_held = False
        self.t_spawn = 0.0
        self.t_grab = None
        self._grab_bonus_consumed = False
        self.model.eq_active[self.connect_eq_id] = 0
        self._spawn_ball_random()
        return self._obs()

    def step(self, a):
        self._apply_action(a)
        for _ in range(4):
            mj_step(model, data)
        self._update_state_machine()
        r = self._compute_reward(a)
        done = self._check_terminal()
        return self._obs(), r, done, {}

    def _update_state_machine(self): ...
    def _compute_reward(self, a): ...
    def _detect_grab(self): ...
    def _detect_release(self): ...
    def _landing_reward(self): ...
```

## Reward total por step

```
r_step = r_smooth + r_effort
       + 1[state=FALLING] · r_approach
       + 1[transition→HELD, first time] · (r_grab_first + r_grab_height + r_grab_speed)
       + 1[state=HELD] · r_hold_dynamics(τ)
       + 1[terminal, !was_held] · (−P_fail)
       + 1[terminal, was_held]  · r_throw_distance
```

## Hyperparams PPO

```
n_envs       = 16 (SubprocVecEnv)
n_steps      = 2048
batch_size   = 256
n_epochs     = 10
lr           = 3e-4 (linear decay a 1e-4)
gamma        = 0.995    # más alto que default por episodios largos
gae_lambda   = 0.95
clip_range   = 0.2
ent_coef     = 0.005
net_arch     = [256, 256]
```

**Normalizar obs con `VecNormalize`. NO normalizar reward** (magnitudes ya balanceadas).

## Cambios necesarios al XML

1. Yellow ball con `<freejoint>`:
```xml
<body name="yellow_ball" pos="0.35 0.25 1.8">
  <freejoint name="ball_free"/>
  <geom name="ball_geom" type="sphere" size="0.04" rgba="1 0.9 0 1"
        mass="0.05" contype="1" conaffinity="1"/>
</body>
```

2. Sites end-effector en ambas pinzas:
```xml
<site name="grip_R" pos="0 0 0.05" size="0.01" rgba="0 1 0 0.5"/>
<site name="grip_L" pos="0 0 0.05" size="0.01" rgba="0 1 0 0.5"/>
```

3. Equality `connect` (inicializado inactivo):
```xml
<equality>
  <connect name="ee_ball_connect" site1="grip_R" body2="yellow_ball"
           active="false" anchor="0 0 0"/>
  ...
</equality>
```

(Equality por lado si entrenamos specialist por brazo; o un solo equality y se elige al spawn cuál usar.)
