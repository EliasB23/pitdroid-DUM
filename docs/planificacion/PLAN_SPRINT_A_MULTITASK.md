# PLAN EJECUTABLE — Multi-task PPO v14 (head + wave + grip)

Generado por agente experto. Para implementar por otro agente.

## Reglas del implementador

- **NO TOCAR** los siguientes archivos/carpetas (los está usando otro proceso):
  - `Scripts/web_remote/`
  - `Scripts/run_interactive.py` (cuando exista)
  - `Scripts/requirements.txt` (línea por línea — la sección web ya está agregada, no rompas)
- **NO LANZAR** training de 6M steps. Solo el smoke test de 50k. El usuario decide cuándo correr el grande.
- **NO MODIFICAR** `Scripts/rl_env.py`, `Scripts/train_ppo.py`, `Scripts/eval_ppo.py` (son del v13, deben quedar intactos como fallback).
- Crear únicamente: `Scripts/rl_env_multitask.py`, `Scripts/train_multitask.py`, `Scripts/eval_multitask.py`.

## Plan técnico

### Paso 1: Archivos a crear/modificar

Crear:
- `Scripts/rl_env_multitask.py` — env nuevo, 7 actuadores, 21 obs.
- `Scripts/train_multitask.py` — wrapper de training con warm-start desde v13.
- `Scripts/eval_multitask.py` — eval con métricas por subtask.

Modificar: ninguno (v13 queda intacto como fallback).

### Paso 2: rl_env_multitask.py

Justificación: archivo nuevo. v13 sigue siendo entregable funcional; aislamos riesgo.

**Imports/constantes** (tope de archivo):
- Copiar todo `rl_env.py`.
- Agregar `WAVE_FREQ_HZ = 0.5`, `WAVE_AMP_SHOULDER = 0.5`, `WAVE_AMP_FOREARM = 0.4`, `WAVE_PHASE = np.pi/4`.
- `GRIP_TARGETS = (0.0, 0.02)`, `GRIP_SWITCH_RANGE = (2.0, 3.0)` (segundos).
- `HOME_POSE_ARM = np.array([0.0, 0.0, 0.0])` (shoulder, forearm, wrist; ajustar si CAD difiere).
- `FLAG_PROB = 0.6`, `W_TRACK = 1.0`, `W_WAVE = 0.6`, `W_GRIP = 0.4`.

**`__init__`**:
- `action_space = Box(-1, 1, (7,))`.
- `observation_space = Box(-inf, inf, (21,))`.
- En `_resolve_ids()`: agregar `act_LeftShoulderArm`, `act_LeftForearm`, `act_LeftWrist`, `act_LeftLever_Slider` via `mujoco.mj_name2id(..., mjOBJ_ACTUATOR, ...)`. Guardar también los `qpos` adr de los joints asociados (`mj_name2id` + `model.jnt_qposadr`).
- Listado `self._policy_actuators = [neck, headbase, headrot, lsh, lfa, lwr, lls]` (7 índices). Resto queda en ctrl=0.

**`reset_model`**:
- Después del reset actual: `self._flags = (np.random.rand(3) < FLAG_PROB).astype(np.float32)`.
- Forzar `if self._flags.sum() == 0: self._flags[np.random.randint(3)] = 1.0`.
- `self._wave_t0 = 0.0`, `self._grip_target = float(np.random.choice(GRIP_TARGETS))`, `self._grip_next_switch = np.random.uniform(*GRIP_SWITCH_RANGE)`.

**`step`**:
- `ctrl = np.zeros(model.nu)`; `ctrl[self._policy_actuators] = action * scale_per_actuator`.
- Mantener `frame_skip` actual.
- Llamar a `_compute_reward()` (Paso 3).

**`_get_obs`**: `np.concatenate([obs_actual_18, self._flags])`.

### Paso 3: Reward multi-task

```
t = data.time
# track (sin cambios, ya tuneado)
r_track = exp(-norm_factor * head_aim_error**2)

# wave
theta_sh  = WAVE_AMP_SHOULDER * sin(2*pi*WAVE_FREQ_HZ*t)
theta_fa  = WAVE_AMP_FOREARM  * sin(2*pi*WAVE_FREQ_HZ*t + WAVE_PHASE)
ref       = [theta_sh, theta_fa, 0.0]
arm_qpos  = data.qpos[[adr_lsh, adr_lfa, adr_lwr]]
if flag_wave: r_wave = exp(-5 * sum((arm_qpos-ref)**2))
else:         r_wave = exp(-5 * sum((arm_qpos-HOME_POSE_ARM)**2))

# grip
if t >= self._grip_next_switch:
    self._grip_target = GRIP_TARGETS[1] if self._grip_target==GRIP_TARGETS[0] else GRIP_TARGETS[0]
    self._grip_next_switch = t + np.random.uniform(*GRIP_SWITCH_RANGE)
slider_q = data.qpos[adr_lls]
r_grip = exp(-50 * (slider_q - self._grip_target)**2)

reward = W_TRACK*flag_track*r_track + W_WAVE*flag_wave*r_wave + W_GRIP*flag_grip*r_grip
```

Guardar componentes en `info` para logging.

### Paso 4: Warm-start desde v13

En `train_multitask.py`, antes de `model.learn()`:

```python
old = PPO.load("runs/ppo_dum_v13/final.zip", device="cpu")
new_sd = model.policy.state_dict()
old_sd = old.policy.state_dict()
for k, v in old_sd.items():
    if k in new_sd and new_sd[k].shape == v.shape:
        new_sd[k] = v
model.policy.load_state_dict(new_sd, strict=False)
```

**Recomendación firme del experto**: mantener `net_arch=[128,128]` para v14 si querés warm-start real, o aceptar que con [256,256] arrancás casi de cero. Plan asume `[128,128]` para aprovechar v13.

### Paso 5: train_multitask.py

- Copiar `train_ppo.py`, reemplazar `DUMHeadTrackingEnv` por `DUMMultitaskEnv`.
- LR schedule lineal SB3:
  ```python
  def lin_sched(initial, final):
      return lambda progress: final + (initial-final)*progress
  learning_rate=lin_sched(3e-4, 1e-4)
  ```
- Hiperparams default: `n_envs=16, n_steps=512, batch_size=256, n_epochs=10, ent_coef=0.005, net_arch=[128,128], gamma=0.99, gae_lambda=0.95`.
- TensorBoard log en `runs/ppo_dum_v14/tb/`.

### Paso 6: Smoke test (OBLIGATORIO antes de 6M)

1. Verificación del env:
   ```bash
   python -c "import sys; sys.path.insert(0,'Scripts'); from rl_env_multitask import DUMMultitaskEnv; e=DUMMultitaskEnv(); o,_=e.reset(); print('obs.shape =', o.shape); print('action.shape =', e.action_space.shape); [e.step(e.action_space.sample()) for _ in range(100)]; print('OK 100 steps')"
   ```
   Debe terminar sin excepción y `obs.shape = (21,)`.

2. Training corto (~5 min): SOLO ESTE TEST, NO LANZAR 6M.
   ```bash
   python Scripts/train_multitask.py --steps 50000 --n-envs 8 --name v14_smoke
   ```
   Revisar log: `ep_rew_mean > 0` y creciendo, `explained_variance > 0`. Si el smoke pasa, dejá el modelo guardado y reportá al usuario para que lance el 6M cuando le convenga.

### Paso 7: NO ejecutar (lo decide el usuario)

Comando para que el usuario corra después:
```bash
python Scripts/train_multitask.py --steps 6000000 --n-envs 16 --n-steps 512 --batch-size 256 --n-epochs 10 --net-arch "128,128" --ent-coef 0.005 --learning-rate 3e-4 --name ppo_dum_v14
```

### Paso 8: eval_multitask.py

- Cargar `final.zip`. Correr N=20 episodios por combinación de flags (8 combos, descartar [0,0,0]).
- Por episodio: acumular `mean_head_err`, `mean_wave_err`, `mean_grip_err`. Imprimir tabla con media±std por flag-combo.
- Video: 4 segmentos forzando flags `[1,0,0]`, `[0,1,0]`, `[0,0,1]`, `[1,1,1]`, 5 s cada uno, `imageio` mp4. Override `_flags` post-reset.

### Paso 9: Riesgos y mitigaciones

- **Reward dominado por r_grip** (exp(-50) muy peakeado): si en smoke `r_grip` aporta < 0.05, bajar a `exp(-25)`.
- **Brazo oscila durante head-tracking**: si `r_track` cae vs v13, subir `W_TRACK` a 1.5 o reducir `W_WAVE/W_GRIP`.
- **`act_LeftShoulderArm` sin convergencia**: verificar en calibración previa que `kp/kv/gear` del shoulder permiten seguir senoides de 0.5 Hz; si no, bajar `WAVE_FREQ_HZ` a 0.3.
- **Warm-start degrada**: si `ep_rew_mean` arranca peor que entrenando from-scratch a 100k, abortar y relanzar sin `load_state_dict`.
- **Equality leva→dedos diverge** con slider cambiando rápido: aumentar `GRIP_SWITCH_RANGE` a (3,5) s.
