# Plan de trainings pendientes

Orden propuesto: v14_head → v15_arm → v16_throw revisit. Cada uno con su rationale
y los parámetros concretos.

---

## v14_head — Tracking 360° (lo más lateral y atrás posible)

### Problema observado
La policy v13_head fue entrenada con target en cono **±75° azimuth**, ±45° elevation.
Cuando el target está más allá (especialmente "atrás" del robot, en el cono +Y world),
la policy NO usa `HeadRotation_joint` (yaw) sino que pone HeadBase perpendicular al
piso — mala estabilidad y look incorrecto. El movimiento natural sería: girar la
cabeza con HeadRot hasta llegar al límite mecánico, ajustar pitch con HeadBase si
hace falta para la altura del target.

### Límite mecánico real (medido en XML, NO 180°)
- `Neck_joint`: ±0.87/+1.22 rad — roll, no es para yaw
- `HeadBase_joint`: ±0.698 rad = **±40°** — pitch (regula altura del lente)
- **`HeadRotation_joint`: ±2.79253 rad = ±160°** — yaw, NO LLEGA A 180°

Confirmado por el usuario: que HeadBase regule la ALTURA del lente NO es problema.
Lo importante es ESTABILIDAD de la cabeza (no perpendicular al piso) y que use
HeadRot para yaw cuando el target lo requiera.

### Cambios propuestos en `Scripts/rl_env.py`
```python
TARGET_CONE_AZIMUTH = np.deg2rad(150)   # antes 75° — 150° deja 10° de margen del max físico 160°
TARGET_CONE_ELEVATION = np.deg2rad(50)  # antes 45° — leve extensión vertical
```

### Reward shaping (sin tocar)
- Mantener `r = w_track * exp(-theta^2/sigma^2)` con sigma=0.4
- Mantener `w_jitter = 0.01` (es lo que da "estabilidad" — penaliza qvel residual)
- Mantener `w_focus_hold = 0.03` para que mantenga el lock

### Setup training
- **Resume desde** `runs/ppo_dum_v13/final.zip`
- **15M steps** (~1.5h estimado con 16 envs CPU)
- Script: `Scripts/train_ppo.py` (verificar args)
- Eval con video después: `Scripts/eval_ppo.py`

### Criterio de éxito
- En 6 targets random azimuth 75-150° (los que v13 no veía), `theta` final < 10°
- HeadRotation usado al menos hasta ±2.0 rad (115°) en esos casos
- HeadBase NO perpendicular al piso (qpos > -0.5) cuando target está atrás
- Sin jitter visible en pose final

---

## v15_arm — Extensión + curriculum static→falling

### Problema observado
La policy v14_arm (50M steps, cap_up=40/cap_out=10cm) tiene catch consistente solo
hasta ~25cm up. En el MAX 40cm, 1/9 en eval determinístico. El brazo se "estira"
pero no llega — min_dist queda ~5-7cm del threshold de 4cm.

### Hipótesis
Falta exposición a configs donde el brazo TIENE QUE estirarse al máximo de forma
controlada. La caída con g=-2.5 da poco tiempo (~0.5s desde 40cm) — la policy
tiene que apurar la extensión y no la perfeccionó.

### Diseño curriculum (sugerencia del usuario)
1. **Primera fase (0-40% del training)**: bola **estática** (`falling_active=False`,
   gravity=0). Cap ramping de 0 hasta `(cap_up=0.50, cap_out=0.15)`. Esto fuerza
   a estirar el brazo SIN time pressure.
2. **Segunda fase (40-100%)**: `falling_active=True`, g=-2.5 (igual que v14).
   Caps quedan en 0.50/0.15.

### Cambios propuestos
- En `rl/envs/grab_env.py`:
  - `CURRICULUM_MAX_UP = 0.50` (era 0.40)
  - `CURRICULUM_MAX_OUT = 0.15` (era 0.10)
- En CLI:
  - `--curriculum-v7d-ramp 0.40` (cap reaches max at 40% del training)
  - `--curriculum-v7e-falling-after 0.40` (falling activa a los 40%)

### Setup training
- **Resume desde** `runs/grab_phase1_v14_long/final.zip`
- **30M steps** (~3.7h estimado)
- Mantener throw enabled + reward shaping de v14
- Eval con video usando configs en distribución nueva

### Criterio de éxito
- En 6 configs static FAR (cap MAX), >=4/6 grabs
- En 6 configs falling MAX (cap_up=50cm, g=-2.5), >=3/6 grabs
- Brazo claramente "estirado" en eval — palma cerca de cap geométrico

---

## v16 — Revisar throw después de v15

### Razonamiento
v15 va a re-entrenar fuerte el catch a las nuevas caps extendidas. Eso puede
DESBALANCEAR el throw (la policy va a invertir más en catch, menos en swing).

Después de v15, evaluar throws en eval para ver si:
- (a) Los throws son comparables a v14 (no degradados) → fin, no hace falta v16
- (b) Throws degradaron → v16 con reward tuning para reforzar swing

### Si se necesita v16
- Resume desde v15
- Boost `THROW_RELEASE_VELOCITY_K`: 500 → 800
- Boost `THROW_LANDING_K`: 1500 → 2000
- 15-20M steps (~2h)

---

## Notas operativas

- **Total budget si se hacen todos**: ~6-7h training + eval+videos
- **PC sin dormir**: configurar antes de cada training largo
- **Resume chain**: v13_head → v14_head, v14_arm → v15_arm
- **Animation engine**: ya tiene autofoco en código (`run_animation_engine.py`,
  bloque `--- 5b. Autofoco`). No cambia entre v13/v14/v15 — sigue cargando lo
  más reciente disponible
