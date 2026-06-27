# Plan A — Mínimo Viable (Garantía de Entrega)

> **Filosofía:** Lo menos posible que produce algo funcional y demostrable en 14 días.
> Sin RL. Sin IK. Sin riesgos. Control directo PD con los gains del grid search.
> El celular opera como interfaz tonta; el PC hace todo el trabajo.

---

## Arquitectura

```
[PC]
  server.py
    ├── MuJoCo headless @ 500Hz
    ├── AnimationEngine (idle + blend)
    ├── AnimationPlayer (clips manuales)
    └── WebSocket server @ 60Hz (puerto 8081)
    └── HTTP server (index.html → puerto 8080)

[Celular — Browser]
  index.html
    ├── Three.js (mallas STL, jerarquía robot)
    ├── WebSocket client (recibe qpos, envía ctrl)
    └── UI: 2 joysticks + 4 botones animación
```

**Protocolo binario** (no JSON):
- Server → Client: `Float32Array` de `nq` valores (qpos)
- Client → Server: `Float32Array` de `nu` valores (ctrl targets)
- Comandos de animación: mensaje texto JSON separado del canal de datos

---

## Tasklist

```
[ ] = pendiente   [x] = completo   [~] = en progreso
```

### FASE 1 — Backend de simulación (Días 1–3)

```
[ ] T01 — Servidor WebSocket base
    ENTREGABLE: server.py carga DUM4.xml, corre mj_step en loop a 500Hz,
    acepta conexiones WebSocket en puerto 8081.
    TEST: conectar con `wscat -c ws://localhost:8081`, recibir bytes.

[ ] T02 — Aplicar configuración óptima del grid search
    ENTREGABLE: función apply_best_configs(model, configs_dict) que lee
    la tabla de kp/kd/damping y la aplica en tiempo de carga.
    TEST: qpos de cada joint se estabiliza en posición neutra sin input.
    INPUT REQUERIDO: tabla best_configs del notebook (CSV o dict).

[ ] T03 — Protocolo de comunicación binario
    ENTREGABLE: serialización/deserialización Float32 en server y client JS.
    Especificación fija:
      Server→Client: [nq floats qpos] + [nu floats actuator_force] = paquete fijo
      Client→Server: [nu floats ctrl]
    TEST: script Python cliente mide RTT promedio. Target: <10ms en loopback.

[ ] T04 — AnimationEngine: capa idle
    ENTREGABLE: clase AnimationEngine con método tick(dt) → dict[str, float].
    Comportamientos implementados:
      - HeadRot + HeadBase: Perlin noise, amplitud 0.02–0.05 rad, f: 0.5–2 Hz
      - BaseHip: sinusoidal 0.05 Hz, amplitud 0.05 rad (weight shifting)
      - HipBody: sinusoidal 0.04 Hz, amplitud 0.03 rad, fase desfasada
    TEST: correr 10s de simulación headless, graficar qpos. Debe oscilar
    suavemente sin divergir.

[ ] T05 — AnimationEngine: blend idle + operador
    ENTREGABLE: función blend_targets(idle, operator, alpha) con rampa
    suave de alpha (0→1 en 0.35s, 1→0 en 0.1s al soltar).
    TEST unitario: verificar valores en t=0, t=0.1, t=0.35.
```

### FASE 2 — Animaciones episódicas (Días 4–5)

```
[ ] T06 — Formato de animación y reproductor
    ENTREGABLE: AnimationClip = List[Tuple[float, Dict[str, float]]]
    Clase AnimationPlayer:
      - play(clip_name): activa clip
      - tick(dt) → dict[str, float]: interpola linealmente entre keyframes
      - is_done() → bool
      - blend_weight() → float: rampa 0→1→0 según duración del clip
    TEST: reproducir clip de 1s, verificar interpolación en t=0, 0.5, 1.0.

[ ] T07 — Biblioteca de clips manuales (4 animaciones)
    ENTREGABLE: archivo clips.py con 4 clips definidos como keyframes:
      'idle_reset':   todos los joints a 0.0 en 1.0s
      'nod_yes':      HeadBase 0→0.3→-0.1→0.3→0 en 1.5s
      'shake_no':     HeadRot 0→0.4→-0.4→0 en 1.2s
      'wave_right':   RightForearm + RightWrist, 2.0s
    TEST: cada clip ejecutado en simulación, qpos grabado y graficado.

[ ] T08 — Integración completa en server loop
    ENTREGABLE: server.py actualizado con loop principal:
      ctrl = blend(idle.tick(dt), operator_input, alpha)
      if player.is_active():
          ctrl = blend(ctrl, player.tick(dt), player.blend_weight())
      data.ctrl[:] = [ctrl[name] for name in actuator_order]
    Comando WebSocket para activar clip: {"cmd": "play", "name": "nod_yes"}
    TEST: enviar comando desde wscat y observar movimiento en simulación.
```

### FASE 3 — Frontend mobile (Días 6–10)

```
[ ] T09 — SPA base con Three.js y carga de mallas STL
    ENTREGABLE: index.html que:
      - Carga las 21 mallas STL con STLLoader (servidas desde /static/)
      - Aplica escala x1000 (compensar coordenadas MuJoCo en mm)
      - Posiciona cada mesh en la jerarquía correcta (padre-hijo según XML)
      - Renderiza a 60fps en Chrome mobile (Samsung S22)
    TEST: abrir en celular, ver robot completo ensamblado correctamente.
    NOTA: usar MeshPhongMaterial diferenciado por grupo:
      torso/cadera → gris oscuro, cabeza → gris claro, brazos → azul grisáceo.

[ ] T10 — Receptor WebSocket y actualización de poses
    ENTREGABLE: función applyMuJoCoState(qpos_buffer) en JS que:
      - Lee Float32Array recibido del servidor
      - Mapea qpos[i] al joint i según tabla hardcodeada (qpos_to_joint_map)
      - Actualiza mesh.rotation o mesh.quaternion según tipo de joint
        (hinge → rotation en eje correcto, slide → position.y/x)
    TEST: mover HeadRot manualmente en server, ver cabeza rotar en Three.js.

[ ] T11 — Joystick virtual y sliders
    ENTREGABLE: UI con:
      - nipplejs: 1 joystick para cabeza (HeadBase Y, HeadRot X)
      - nipplejs: 1 joystick para torso (HipBody Y, BaseHip X)
      - 4 sliders HTML para: LeftForearm, RightForearm, LeftWrist, RightWrist
    Output: envía Float32Array de ctrl targets al servidor cada 50ms.
    TEST: mover joystick en celular, ver robot moverse en Three.js en <100ms.

[ ] T12 — Botones de animación episódica en UI
    ENTREGABLE: 4 botones mapeados a los clips de T07.
    Estado visual: botón deshabilitado (opacity 0.4) mientras clip activo.
    Envía JSON: {"cmd": "play", "name": "<clip_name>"}
    TEST: presionar 'nod_yes', robot asiente, botón se rehabilita al terminar.
```

### FASE 4 — Integración y cierre (Días 11–14)

```
[ ] T13 — Servidor HTTP integrado en server.py
    ENTREGABLE: server.py sirve archivos estáticos en puerto 8080
    (index.html + STLs + JS). Usando http.server o aiohttp.
    TEST: abrir http://<IP_PC>:8080 desde celular en red local, ver UI.

[ ] T14 — Medición de latencia y ajuste
    ENTREGABLE: timestamp en cada paquete Float32 (primer elemento reservado).
    Log automático de RTT p50 y p95 durante 60s.
    Si RTT p95 > 50ms: reducir frecuencia de envío de 60Hz a 30Hz.
    TEST: log muestra RTT p95 < 50ms en WiFi local.

[ ] T15 — Iluminación y materiales finales
    ENTREGABLE: escena Three.js con:
      - AmbientLight (0.4 intensity)
      - DirectionalLight desde arriba-frente (0.8 intensity)
      - Sombras activadas en el plano base
    TEST visual: robot se ve legible en pantalla de celular con brillo medio.

[ ] T16 — Test de integración final
    ENTREGABLE: sesión de 10 minutos sin intervención con log automático.
    Checklist verificado:
      [x] Conexión WebSocket estable durante 10 min
      [x] Animaciones ejecutan sin glitches
      [x] Control responsivo (RTT < 50ms)
      [x] qpos no diverge (max(|qpos|) < joint_range_max para cada joint)
      [x] Renderer mantiene >55fps en celular
```

---

## Dependencias externas requeridas

```
Python (PC):
  mujoco >= 3.0
  websockets
  aiohttp
  noise          # Perlin noise para AnimationEngine
  numpy

JavaScript (Browser):
  three.js r150+
  three/examples/jsm/loaders/STLLoader.js
  nipplejs        # joystick virtual
```

---

## Estructura de archivos esperada

```
DUM4_sim/
├── server.py              # entry point, WebSocket + HTTP
├── animation_engine.py    # AnimationEngine + AnimationPlayer
├── clips.py               # biblioteca de clips keyframe
├── best_configs.py        # kp/kd/damping del grid search
├── Cuerpo/
│   └── DUM4.xml
│   └── meshes/            # 21 archivos STL originales
└── static/
    ├── index.html
    ├── main.js            # Three.js scene + WebSocket client
    ├── ui.js              # joysticks + botones
    └── meshes/            # copia de STLs servida por HTTP
```

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Mapeo qpos→Three.js incorrecto | Alta | Empezar con 1 joint, validar visualmente antes de mapear todos |
| STLs pesadas en mobile | Media | Si carga >5s, simplificar mallas con trimesh antes de servir |
| RTT > 50ms en WiFi | Baja | Reducir frecuencia a 30Hz, comprimir payload con pako.js |
| Slider/pinza no resuelto | Alta | Excluir act_LeftLever_Slider y act_RightLever_Slider del control por ahora |

---

*Tiempo total estimado: 14 días. Probabilidad de entrega completa: ~90%.*
