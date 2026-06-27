# Plan B — Completo con RL y Animation Engine de Producción

> **Filosofía:** Implementar la misma arquitectura conceptual de los papers de Disney
> (BD-X y Olaf) adaptada a DUM4: torso fijo, presupuesto de consumer hardware,
> 14 días de desarrollo con Claude Code.
>
> **Advertencia crítica — GTX 1650 (4GB, 896 CUDA cores):**
> Isaac Gym no soporta esta GPU oficialmente. La ruta RL es MuJoCo MJX (JAX/CUDA).
> Con 4GB VRAM el máximo práctico es ~128 entornos paralelos.
> Estimación de throughput: 5k–15k steps/segundo.
> Para convergencia de política de postura (~500k steps): 30–100 minutos.
> **T00a es un gate — si el benchmark falla, el plan colapsa a Plan A el Día 1.**

---

## Arquitectura completa

```
[PC — servidor]
  server.py (asyncio)
    ├── SimulationPod: MuJoCo @ 500Hz (datos físicos)
    ├── PolicyPod: inference RL @ 50Hz (posición setpoints → PD controllers)
    ├── AnimationEngine:
    │     ├── BackgroundAnimator (Perlin + osciladores, siempre activo)
    │     ├── TriggeredLayer (clips con blend-in/out)
    │     └── JoystickMapper (input operador → offsets de pose)
    ├── WebSocket server @ 60Hz (puerto 8081)
    └── HTTP server (SPA → puerto 8080)

[Celular — Browser]
  index.html
    ├── Three.js (mallas GLTF comprimidas, jerarquía correcta)
    ├── WebSocket client (recibe xquat bodies, envía joystick + comandos)
    └── UI: 2 joysticks + 8 botones animación + panel de estado
```

**Protocolo binario:**
- Server → Client: `Float32Array` de `nbody*4` quaternions (xquat de todos los bodies)
  Nota: usar xquat en lugar de qpos evita integrar la cadena cinemática en JS.
- Client → Server: `Float32Array` de 4 valores (joy_x, joy_y, joy2_x, joy2_y)
- Comandos: JSON por canal separado

---

## Tasklist

```
[ ] = pendiente   [x] = completo   [~] = en progreso
```

### FASE 0 — Verificación de infraestructura (Día 1, BLOQUEANTE)

```
[ ] T00a — Benchmark GPU para MJX
    ENTREGABLE: script benchmark_mjx.py que:
      1. Instala jax[cuda12] + mujoco-mjx
      2. Vectoriza DUM4 con N=32, 64, 128 entornos en paralelo
      3. Corre 1000 steps y reporta steps/segundo por configuración
      4. Mide uso de VRAM con nvidia-smi durante el benchmark
    CRITERIO DE PASE: N=64 entornos a >3000 steps/s sin OOM
    CRITERIO DE FALLO: OOM con N=32 o <1000 steps/s con N=64
    SI FALLA: plan colapsa a Plan A. Notificar inmediatamente.
    TEST: log final imprime "MJX VIABLE: N=X a Y steps/s" o "MJX NO VIABLE".

[ ] T00b — Pipeline de optimización de mallas
    ENTREGABLE: script optimize_meshes.py que:
      1. Carga las 21 STL con trimesh
      2. Simplifica a 30% de triángulos con quadric decimation
      3. Exporta como GLTF con compresión Draco
      4. Genera reporte: tamaño original vs final, error de geometría (Hausdorff)
    TARGET: total < 1.5MB, error Hausdorff < 0.5mm
    TEST: cargar en Three.js mobile, tiempo de carga < 2s en WiFi local.
```

### FASE 1 — Entorno RL (Días 1–4)

```
[ ] T01 — Entorno Gymnasium base para DUM4
    ENTREGABLE: clase DUM4StandingEnv(gymnasium.Env) con:
      observation_space: Box(shape=(nq + nv + nu,)) — posición + velocidad + fuerza
      action_space: Box(shape=(nu,), low=-1, high=1) — normalizado, se desnormaliza
                    con ctrlrange de cada actuador
      step(action):
        - desnormalizar acción
        - asignar data.ctrl
        - mj_step x4 (substeps para estabilidad)
        - calcular reward
        - retornar obs, reward, terminated, truncated, info
      reset(seed, options):
        - posición neutra + ruido N(0, 0.02) en qpos
        - velocidad = 0
    TEST: gymnasium.utils.env_checker.check_env(DUM4StandingEnv())
    sin errores. 1000 steps con acción aleatoria sin crash.

[ ] T02 — Función de reward compuesta
    ENTREGABLE: módulo rewards.py con 4 términos y sus pesos:

      r_track(q, q_target, w=10.0):
        exp(-10.0 * mean((q - q_target)²))
        Rango: [0, 1]. 1 = pose perfecta.

      r_smooth(tau, w=0.001):
        -mean(tau²)
        Penaliza torque excesivo. Escala según actuador.

      r_limits(q, q_dot, q_min, q_max, gamma=20.0, margin=0.05, w=0.5):
        CBF de joint limits (igual que paper Olaf, Ec. 7-8).
        Penaliza cuando q se acerca a límites con velocidad hacia ellos.

      r_action_rate(a, a_prev, w=0.5):
        -mean((a - a_prev)²)
        Promueve suavidad de acción entre steps.

      r_total = r_track + r_smooth + r_limits + r_action_rate

    TEST unitario: cada término retorna escalar en rango esperado
    para (q=q_neutral, action=zeros) y (q=random, action=random).

[ ] T03 — Vectorización con MJX
    ENTREGABLE: clase VecDUM4Env que usa jax.vmap sobre DUM4StandingEnv.
    Implementación:
      - MjModel.from_xml_path → modelo base
      - mjx.put_model(model) → modelo en GPU
      - jax.vmap(mjx.step)(model, batch_of_data) → N steps en paralelo
      - Wrapper que expone interfaz gymnasium VecEnv estándar
    N configurable, default N=64 para GTX 1650.
    TEST: VecDUM4Env(n_envs=64).step(random_actions) corre en <100ms.
    Benchmark: reportar steps/segundo al inicializar.

[ ] T04 — Poses de referencia para imitation reward
    ENTREGABLE: archivo reference_poses.json con 4 poses:
      "neutral":    todos los joints en 0.0
      "attention":  HeadBase: 0.3, HeadRot: 0, HipBody: 0.1, brazos semiflexionados
      "relaxed":    HipBody: -0.05, brazos ligeramente caídos (ver valores en XML)
      "scan":       HeadRot: 0.4, HeadBase: 0.2 (mirando a un lado y arriba)
    Formato: {"pose_name": {"joint_name": float, ...}}
    DUM4StandingEnv.reset(options={"target_pose": "attention"}) carga la pose.
    TEST: visualizar cada pose en MuJoCo viewer antes de usarla en RL.
```

### FASE 2 — Entrenamiento RL (Días 4–8)

```
[ ] T05 — Training loop con PPO (stable-baselines3)
    ENTREGABLE: script train.py con configuración:
      PPO(
        policy = "MlpPolicy",
        env = VecDUM4Env(n_envs=64),
        n_steps = 512,        # reducido por VRAM limitada
        batch_size = 256,
        n_epochs = 10,
        learning_rate = 3e-4,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        ent_coef = 0.0,
        vf_coef = 0.5,
        max_grad_norm = 1.0,
        policy_kwargs = dict(net_arch=[512, 512, 256])
      )
    Callbacks:
      - CheckpointCallback: guardar cada 25k steps en /checkpoints/
      - EvalCallback: evaluar en env sin vectorizar cada 10k steps
      - TensorBoard: log de r_track, r_smooth, r_limits, r_action_rate por separado
    TEST: después de 10k steps, reward_mean debe ser mayor que en step 0.
    Comando: python train.py --pose neutral --steps 500000 --name standing_v1

[ ] T06 — Política standing (postura neutra robusta)
    ENTREGABLE: política entrenada standing_policy.zip con:
      - MAE < 0.05 rad en postura neutra
      - Estable bajo perturbaciones N(0, 0.1) en qpos inicial
      - No viola joint limits en rollout de 500 steps
    Evaluación: script eval_policy.py --policy standing_policy.zip
    que corre 10 episodios y reporta MAE por joint.
    TIEMPO ESTIMADO: 30–60 min en GTX 1650 para 500k steps.

[ ] T07 — Políticas de pose objetivo (3 adicionales)
    ENTREGABLE: 3 políticas entrenadas con target_pose distinto:
      attention_policy.zip:  target = pose "attention"
      relaxed_policy.zip:    target = pose "relaxed"
      scan_policy.zip:       target = pose "scan"
    Mismo criterio de evaluación que T06.
    NOTA: estas políticas pueden entrenarse en paralelo si la VRAM lo permite.
    Si no: entrenar secuencialmente. Tiempo total estimado: 2–4 horas.

[ ] T08 — Policy blending para transiciones suaves
    ENTREGABLE: clase PolicyBlender en policy_blender.py:
      __init__(policies: Dict[str, PPO])
      blend(obs, weights: Dict[str, float]) → action:
        actions = {name: policy.predict(obs)[0] for name, policy in policies}
        return sum(w * actions[name] for name, w in weights.items()) / sum(weights.values())
      transition_to(target_policy, duration=0.35):
        rampa lineal de pesos durante `duration` segundos
    TEST: blend(obs, {"standing": 0.5, "attention": 0.5}) produce acción
    intermedia entre ambas políticas. Transición de 0.35s no produce
    discontinuidad en acción (verificar con gráfico).
```

### FASE 3 — Animation Engine completo (Días 5–9, paralelo con FASE 2)

```
[ ] T09 — Módulo de osciladores procedurales
    ENTREGABLE: módulo procedural.py con:
      perlin1d(t, octaves=4, persistence=0.5, lacunarity=2.0) → float
        Usando biblioteca `noise`. Normalizado a [-1, 1].

      class OscillatorBank:
        __init__(joints: List[str], freq_range, amp_range, phase_random=True)
        tick(dt) → Dict[str, float]
        Cada joint tiene frecuencia y fase independientes sorteadas en __init__.

      class SaccadeGenerator:
        Genera micro-movimientos para HeadRot y HeadBase:
        - En reposo: Perlin noise de baja amplitud (0.01–0.03 rad)
        - Sácada: cada 3–8s (aleatoriamente), movimiento rápido (0.02–0.08 rad)
          con duración 50–150ms seguido de fijación por 1–4s
        tick(dt) → Dict[str, float]

      class LensAutofocus:
        Simula hunting de autofocus para LenteExt_joint (slide):
        - En reposo: pequeña oscilación amortiguada aleatoria
        - Cada 10–20s: pulso de búsqueda (oscilación 3–5 ciclos) que converge
        tick(dt) → float (posición del slide)

    TEST: señal de cada oscilador durante 30s graficada, verificar que
    amplitudes y frecuencias caen dentro de los rangos definidos.

[ ] T10 — BackgroundAnimator (Capa 1)
    ENTREGABLE: clase BackgroundAnimator:
      Integra SaccadeGenerator + OscillatorBank + LensAutofocus.
      Comportamientos activos permanentemente:
        - HeadRot, HeadBase: SaccadeGenerator
        - BaseHip: OscillatorBank, f=0.04–0.06Hz, amp=0.04–0.06 rad
        - HipBody: OscillatorBank, f=0.03–0.05Hz, amp=0.02–0.04 rad, fase desfasada
        - act_LeftLever_Slider, act_RightLever_Slider: excluidos (bug no resuelto)
        - LenteExt_joint: LensAutofocus
      tick(dt) → Dict[str, float]  (ctrl targets normalizados en ctrlrange)
    TEST: correr 60s de simulación solo con BackgroundAnimator,
    verificar que ningún joint viola sus límites.

[ ] T11 — Sistema de clips con interpolación cúbica
    ENTREGABLE: módulo clips_engine.py:
      AnimationClip:
        keyframes: List[Tuple[float, Dict[str, float]]]
        duration: float (último timestamp)
        interp: 'linear' | 'cubic'  (cubic = CubicSpline de scipy)
        sample(t) → Dict[str, float]  (interpola en tiempo t)

      AnimationPlayer:
        load_clip(name, clip: AnimationClip)
        play(name) → None
        stop() → None
        tick(dt) → Tuple[Dict[str, float], float]
          retorna (ctrl_targets, blend_weight)
          blend_weight: 0→1 en Tα=0.35s al inicio, 1→0 en Tβ=0.1s al final
        is_active() → bool

    TEST: clip de 2s con interpolación cúbica, muestrear en 100 puntos,
    verificar que posición y velocidad son continuas (sin saltos).

[ ] T12 — Biblioteca de clips (8 animaciones) generada con optimización de jerk
    ENTREGABLE: script generate_clips.py que produce clips con trayectorias
    mínimo-jerk usando scipy.optimize.minimize:

      Objetivo: minimizar integral de ||d³q/dt³||² sujeto a:
        - q(0) = q_neutral (posición inicial)
        - q(T) = q_target  (posición final)
        - q_dot(0) = q_dot(T) = 0  (arranque y parada suaves)
        - q_min ≤ q(t) ≤ q_max    (joint limits)

    Clips a generar:
      'idle_reset':    todos → 0, T=1.5s
      'nod_yes':       HeadBase 0→0.35→0, T=1.8s (2 ciclos)
      'shake_no':      HeadRot 0→0.45→-0.45→0, T=2.0s
      'look_around':   HeadRot + HeadBase trayectoria circular, T=3.0s
      'wave_right':    RightForearm + RightWrist, T=2.5s
      'wave_left':     LeftForearm + LeftWrist, T=2.5s
      'attention':     transición a pose "attention", T=1.5s
      'scan':          secuencia de 3 posiciones de cabeza, T=4.0s

    Guardados como JSON en /clips/. Cargados por AnimationPlayer.
    TEST: cada clip visualizado en MuJoCo viewer antes de integrar.

[ ] T13 — TriggeredLayer (Capa 2) con blend suave
    ENTREGABLE: clase TriggeredLayer:
      Estados: IDLE → BLEND_IN → ACTIVE → BLEND_OUT → IDLE
      activate(clip_name): inicia transición
      tick(dt) → Tuple[Dict[str, float], float]
        retorna (targets_del_clip, blend_weight)
        blend_weight: rampa según estado actual
      Duración de rampas: Tβ=0.1s (show functions), Tα=0.35s (body)
        (igual que Disney paper, Ec. 9-10)
    TEST: activar clip, verificar que blend_weight sigue la rampa correcta
    en cada estado con gráfico de blend_weight vs tiempo.

[ ] T14 — JoystickMapper (Capa 3)
    ENTREGABLE: clase JoystickMapper:
      Recibe: joy1=(x,y), joy2=(x,y) en [-1, 1]
      Mapping no-lineal (cuadrático para precisión en centro):
        joy1_y → HeadBase offset (rango ±0.3 rad)
        joy1_x → HeadRot offset  (rango ±0.4 rad)
        joy2_y → HipBody offset  (rango ±0.2 rad)
        joy2_x → BaseHip offset  (rango ±0.1 rad)
      get_offsets(joy1, joy2) → Dict[str, float]
      Nota: offsets se suman a la salida de la política RL,
      no reemplazan — igual que Disney "additive offsets".
    TEST: joy1=(1,0) produce HeadRot en máximo positivo,
    joy1=(0,0) produce HeadRot=0.

[ ] T15 — AnimationEngine: integración de 3 capas
    ENTREGABLE: clase AnimationEngine que combina todo:
      __init__(background, triggered, joystick, policy_blender)
      tick(dt, joy1, joy2, command=None) → Dict[str, float]:
        1. bg_targets = background.tick(dt)
        2. triggered_targets, triggered_weight = triggered.tick(dt)
        3. blended = lerp(bg_targets, triggered_targets, triggered_weight)
        4. joy_offsets = joystick.get_offsets(joy1, joy2)
        5. policy_action = policy_blender.blend(obs, active_weights)
        6. final = apply_policy_with_offsets(policy_action, blended, joy_offsets)
      handle_command(cmd: dict): despacha comandos de play/policy_switch
    TEST de integración: 60s de tick() con comandos aleatorios.
    Verificar que ningún target viola ctrlrange de su actuador.
```

### FASE 4 — Frontend de producción (Días 8–12)

```
[ ] T16 — Escena Three.js con mallas GLTF y jerarquía cinemática
    ENTREGABLE: index.html con escena que replica la jerarquía del XML:
      Base_link (fijo)
        └── BaseHip_link (pivot en BaseHip_joint.pos)
              └── FullBody_link (pivot en HipBody_joint.pos)
                    ├── [cadena izquierda: hombro → antebrazo → muñeca → dedos]
                    ├── [cadena derecha: ídem]
                    └── [cuello → HeadBase → HeadRot → lente]
    Cada Object3D tiene su pivot point en el origen del joint correspondiente.
    Los valores pos y euler del XML se usan para posicionar los pivots.
    TEST: mover BaseHip_joint manualmente en Three.js console, verificar
    que toda la cadena downstream se mueve correctamente.

[ ] T17 — Mapeo xquat → Three.js (world frame directo)
    ENTREGABLE: función updateFromMuJoCoWorldFrame(xquat_buffer):
      En lugar de integrar qpos joint por joint, usa data.xquat:
        - xquat[body_id] = quaternion del body en world frame
        - Three.js: mesh.quaternion.set(x, y, z, w)
          (MuJoCo orden: [w,x,y,z] → Three.js: [x,y,z,w])
      Tabla de mapeo body_id → mesh_name hardcodeada.
    Server envía: Float32Array de shape (nbody * 4,) con xquat.
    TEST: hover de un joint en MuJoCo produce rotación correcta en Three.js
    para todos los joints de la cadena.

[ ] T18 — UI mobile completa
    ENTREGABLE: UI responsive en portrait mode:
      - 2 joysticks virtuales (nipplejs) en la parte inferior
      - 8 botones de animación en grid 2x4 en la parte superior
      - Indicador de conexión (verde/rojo) en esquina superior derecha
      - Slider de "alpha" para blend idle/operador (horizontal, top)
      - Panel colapsable con métricas: FPS, RTT, policy activa
    Todos los elementos accesibles con un dedo en S22.
    TEST: operación con una mano durante 5 minutos sin perder elementos.

[ ] T19 — IK simplificado para control de gaze por touch
    ENTREGABLE: modo alternativo de control donde tocar la pantalla
    dirige la "mirada" de la cabeza hacia ese punto (en plano de pantalla):
      1. Tocar punto (screen_x, screen_y) → world_direction via raycasting
      2. Calcular ángulos HeadBase y HeadRot necesarios con IK analítico
         (el robot es torso fijo, la cadena cabeza tiene 2 DOF → IK cerrada)
      3. Enviar como joystick offset vía WebSocket
    IK analítica (no iterativa): no hay riesgo de divergencia.
    TEST: tocar cuatro esquinas de la pantalla, robot mira en esas direcciones.
    NOTA: si la implementación toma >2 días, reemplazar con joystick normal (T11 del Plan A).
```

### FASE 5 — Integración RL + Animation Engine (Días 11–14)

```
[ ] T20 — Policy inference en server loop @ 50Hz
    ENTREGABLE: server.py reemplaza ctrl directo por inference de política:
      - PolicyBlender cargado con todas las políticas entrenadas
      - Loop @ 50Hz: obs = get_obs(data) → action = blender.blend(obs, weights)
      - action son joint position setpoints → data.ctrl
      - Loop de simulación @ 500Hz corre independiente (action se mantiene
        entre steps de política, como first-order hold del paper de Disney)
    TEST: simulación corre 60s con standing_policy activa.
    Verificar: MAE < 0.05 rad en postura neutra durante toda la sesión.

[ ] T21 — Policy switching via WebSocket
    ENTREGABLE: comando {"cmd": "policy", "name": "attention"} activa
    transición suave de 0.35s via PolicyBlender.transition_to().
    UI: los 8 botones de animación pueden ser 4 clips + 4 policy switches.
    TEST: switch durante operación, graficar qpos durante la transición.
    No debe haber discontinuidad (salto) en ningún joint.

[ ] T22 — Grabación y replay de sesiones
    ENTREGABLE: server acepta comandos {"cmd": "record_start"} / {"cmd": "record_stop"}.
    Graba: timestamps, qpos, xquat, ctrl, policy_weights, joystick_input.
    Guardado como NPZ en /recordings/<timestamp>.npz.
    Script replay.py --file <recording.npz>:
      reproduce la sesión en simulación y en el frontend (headless replay).
    Uso principal: generar nuevos clips de animación a partir de performance
    del operador, que luego se optimizan con generate_clips.py (T12).
    TEST: grabar 30s, reproducir, verificar que qpos coincide sample a sample.

[ ] T23 — Test de integración y métricas finales
    ENTREGABLE: script eval_integration.py que corre 5 minutos automáticos:
      Métricas reportadas:
        - MAE de joint tracking por política (target: < 0.05 rad)
        - RTT WebSocket p50 y p95 (target: p95 < 50ms)
        - FPS del renderer en celular (target: > 55fps)
        - Uso de CPU del servidor (target: < 80%)
        - Uso de VRAM durante inference (target: < 3GB para GTX 1650)
        - Violaciones de joint limits (target: 0)
      Reporte en /reports/integration_<timestamp>.json.
    TEST final: todas las métricas en target. Si alguna falla, identificar cuál
    tarea es responsable y registrar en el reporte.
```

---

## Dependencias externas requeridas

```
Python (PC):
  mujoco >= 3.0
  mujoco-mjx          # MJX para vectorización GPU
  jax[cuda12_pip]     # JAX con soporte CUDA 12
  stable-baselines3
  gymnasium
  noise               # Perlin noise
  scipy               # CubicSpline para clips
  trimesh             # optimización de mallas
  draco               # compresión de mallas (via trimesh)
  websockets
  aiohttp
  numpy
  tensorboard

JavaScript (Browser):
  three.js r150+
  three/examples/jsm/loaders/GLTFLoader.js
  three/examples/jsm/loaders/DRACOLoader.js
  nipplejs
  pako.js             # compresión opcional si RTT > 50ms
```

---

## Estructura de archivos esperada

```
DUM4_sim/
├── server.py                  # entry point principal
├── simulation_pod.py          # MuJoCo loop @ 500Hz
├── policy_pod.py              # RL inference @ 50Hz + PolicyBlender
├── animation_engine.py        # AnimationEngine (3 capas)
├── procedural.py              # Perlin, OscillatorBank, SaccadeGenerator, LensAutofocus
├── clips_engine.py            # AnimationClip + AnimationPlayer
├── rewards.py                 # funciones de reward RL
├── vec_env.py                 # VecDUM4Env con MJX
├── best_configs.py            # kp/kd/damping del grid search
├── train.py                   # script de entrenamiento RL
├── eval_policy.py             # evaluación de política individual
├── eval_integration.py        # test de integración completo
├── generate_clips.py          # generación de clips con mínimo-jerk
├── optimize_meshes.py         # STL → GLTF+Draco
├── benchmark_mjx.py           # T00a — gate de viabilidad RL
├── reference_poses.json       # 4 poses de referencia
├── checkpoints/               # políticas guardadas durante entrenamiento
├── policies/                  # políticas finales
│   ├── standing_policy.zip
│   ├── attention_policy.zip
│   ├── relaxed_policy.zip
│   └── scan_policy.zip
├── clips/                     # clips de animación en JSON
│   └── *.json
├── recordings/                # sesiones grabadas
├── reports/                   # reportes de evaluación
├── Cuerpo/
│   ├── DUM4.xml
│   └── meshes/                # STL originales
└── static/
    ├── index.html
    ├── main.js                # Three.js scene
    ├── ui.js                  # joysticks + botones + IK touch
    └── meshes/                # GLTF comprimidos servidos por HTTP
```

---

## Plan de contingencia por tarea

| Tarea | Si falla o toma demasiado | Fallback |
|-------|--------------------------|---------|
| T00a (MJX benchmark) | Cualquier fallo | Colapso total a Plan A |
| T03 (VecEnv MJX) | OOM con N=64 | Reducir a N=32 o usar multiprocessing CPU |
| T06-T08 (políticas RL) | No convergen en tiempo | Usar PD directo con best_configs (Plan A fallback parcial) |
| T12 (clips mínimo-jerk) | scipy.optimize no converge | Reemplazar con keyframes manuales (T07 del Plan A) |
| T19 (IK touch) | Implementación compleja | Reemplazar con joystick estándar (T11 del Plan A) |
| T22 (grabación) | Overhead de I/O en loop | Grabar en proceso separado con queue asyncio |

---

## Distribución temporal por fase

```
Día  1:  T00a (GATE), T00b, T01
Día  2:  T01 (cont), T02, T03
Día  3:  T03 (cont), T04
Día  4:  T04 (cont), T05, T09
Día  5:  T05 (entrenamiento T06), T09 (cont), T10
Día  6:  T06 (eval), T07 (entrenamiento), T10 (cont), T11
Día  7:  T07 (cont), T08, T11 (cont), T12
Día  8:  T08, T12 (cont), T13, T16
Día  9:  T13, T14, T15, T16 (cont)
Día 10:  T15, T17, T18
Día 11:  T17 (cont), T18 (cont), T19, T20
Día 12:  T19 (cont), T20 (cont), T21
Día 13:  T21, T22, T23 (parcial)
Día 14:  T23 (completo), buffer para bugs críticos
```

---

## Diferencias clave respecto a los papers de Disney

| Aspecto | BD-X / Olaf | DUM4 (Plan B) |
|---------|-------------|---------------|
| Hardware RL | RTX 4090, Isaac Gym/Sim | GTX 1650, MuJoCo MJX |
| Entornos paralelos | 8192 | 64–128 |
| Tiempo de entrenamiento | 2 días por política | 30–100 min por política |
| Locomoción | Bipedal dinámica | Torso fijo (más simple) |
| Imitation reward | Mocap/Maya animations | Poses JSON + mínimo-jerk |
| Show functions | Antennas, LEDs, audio | Ojo (LenteExt_joint), cabeza |
| Puppeteering | Steam Deck físico | Browser en celular |
| Thermal modeling | Sí (Olaf) | No (motores DS3218 no reportan temperatura) |

---

*Tiempo total estimado: 14 días. Probabilidad de entrega completa: ~55–65%.*
*Probabilidad de entrega parcial funcional (sin RL o sin IK touch): ~85%.*
*El gate T00a el Día 1 elimina el riesgo de descubrir la inviabilidad del RL tarde.*
