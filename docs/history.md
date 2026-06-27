# History — Sprint Plan B (RL Head-Tracking)

Bitácora de acciones de alto nivel. Útil para reconstruir el desarrollo en la presentación final.

## 2026-05-15 — Análisis inicial y benchmarks

- Analicé estructura del proyecto: modelo MJCF, mallas STL, scripts de calibración existentes.
- Leí el documento del proyecto (`Proyecto CREACIÓN DE ROBOT ANIMATRÓNICO 202604.docx`) para entender hipótesis, alcance y estado de la calibración.
- Generé `CLAUDE.md` con objetivos del proyecto, stack técnico, convenciones MJCF y reglas de trabajo.
- Creé el agente especializado `mujoco-rl-expert` en `~/.claude/agents/` para futuras consultas técnicas sobre MuJoCo y RL.
- Escribí `Scripts/benchmarks.py` con 5 pruebas: estabilidad, sweep de integradores, throughput, convergencia actual de calibración y auditoría MJX.
- Ejecuté los 5 benchmarks; identifiqué 2 bugs en el XML (`ctrlrange` de los `Lever_Slider`, joint range de `HeadRotation_joint`) y 6 actuadores con calibración insuficiente.
- Elaboré dos planes (Plan A conservador, Plan B con RL) y recomendé Plan B con regla de corte temprano si la policy no converge.

## 2026-05-16 — Inicio Plan B

- Verifiqué que los fixes del XML (joint range de `HeadRotation_joint`, `ctrlrange` de los dos `Lever_Slider`) hayan sido aplicados manualmente por el usuario.
- Detecté dos cambios faltantes y los apliqué:
  - Agregué `<option integrator="implicit" timestep="0.005" />` al XML (mejor respuesta del PID y 2.5× throughput).
  - Ajusté `ctrlrange` y `kv` de `act_HeadRot` para que el espacio de control corresponda al rango físico del joint.
- Re-ejecuté B1 + B4 con el modelo actualizado: estabilidad OK; 8/13 actuadores pasan el escalón. Quedan 5 para recalibrar más adelante.
- Creé este `history.md` como bitácora del sprint.
- Detecté que el `LenteExt_joint` (slide del lente interno, para enfocar) no tenía actuador asociado. Agregué `act_LenteExt` al XML para poder controlarlo.
- Agregué un cuerpo `mocap` llamado `target` al XML — esfera roja sin masa ni colisión, cuya posición se setea desde el código del env. Sirve como objetivo visual del head-tracking y para que el agente "vea" dónde está el target.

### Diseño de la función de reward (charlado y acordado)

Tarea: el robot orienta su cabeza para mirar a un punto objetivo. Solo se controlan `act_Neck`, `act_HeadBase`, `act_HeadRot`. Más adelante, `act_LenteExt` para el "enfoque" animatrónico.

Reward total por step (sin LaTeX):

    r_t = r_track + r_smooth + r_effort + r_alive − r_jitter

Componentes:

- **r_track = w1 · exp(−θ² / σ²)**
  θ es el ángulo entre el vector "mirando" de la cabeza y el vector cabeza→target. σ ≈ 0.1 rad (5.7°).
  Recompensa acercar la cabeza al target. Exponencial en vez de cuadrática porque satura: no castiga proporcional a la distancia cuando el target reaparece lejos, evita "pánico" en la policy.
  Peso: w1 = 1.0 (referencia).

- **r_smooth = − w2 · ||a_t − a_{t−1}||²**
  Castiga cambios bruscos en el comando. Higiene sim-to-real (servos reales no toleran chatter) y wow factor visual (cabeza animatrónica suave, no robotínica).
  Peso: w2 = 0.05 (a tunear).

- **r_effort = − w3 · ||a_t||²**
  Castiga magnitud absoluta del comando. Evita que el agente mande ctrl=±max aunque ya esté en target.
  Peso acordado: **w3 = 0.0075** (reducido desde 0.01 porque el robot ya hace esfuerzo permanente compensando gravedad y miembros sueltos).

- **r_alive = + w4** (constante por step)
  Bonus por seguir corriendo. En esta tarea no aplica: r_track ≥ 0 y no hay terminación temprana, así que w4 = 0.

- **r_jitter = − w5 · ||q̇_head||²**
  Penalización por velocidad alta de los joints de la cabeza. Complementa r_smooth.
  Peso: w5 = 0 inicial; se activa solo si vemos oscilaciones físicas en eval.

Observaciones que recibe la red (11 dims):

- `qpos` de [Neck, HeadBase, HeadRot] — 3 dims
- `qvel` de [Neck, HeadBase, HeadRot] — 3 dims
- Vector target→cabeza expresado en frame local de la cabeza — 3 dims
- Acción anterior a_{t−1} — 2 dims (necesaria para r_smooth)

Reset: pose neutra (joints a 0), target sampleado uniformemente en cono frontal ±60° azimut × ±30° elevación, distancia 0.3–1.0 m. Después del reset, ~50 steps de settling con ctrl=0 antes de devolver la primera observación.

Terminación: nunca (tarea continua). Truncación: 500 steps (10 s a 50 Hz de control).

Decisiones del usuario:
1. Empezar con target estático por episodio. Escalar a dinámico solo si converge rápido.
2. Mirar con el eje de la lente (precisión "ojo apuntando al target"), no del cuello.
3. r_effort con peso 0.0075 (más permisivo).
4. Visualizar target (esfera roja, ya agregada al XML) y vector mirando.
5. Agregar comportamiento extra de "enfoque animatrónico": cuando la cabeza alcanza el target por primera vez en el episodio, el lente interno (`act_LenteExt`) se mueve adelante-atrás simulando un enfoque óptico.

Decisión adicional sobre el enfoque: implementación **Opción B (hook determinista)**, disparado la primera vez que `θ < 0.07 rad` (≈ 4°, equivalente a r_track > 0.6) después de un settling inicial de 10 steps. Umbral justificado en función del σ del exponencial de `r_track`. El hook ejecuta una secuencia hard-coded de 2 ciclos cosenoidales sobre `act_LenteExt` y luego se inhabilita hasta el próximo reset.

### Implementación del environment y training

- Instalé el stack RL: `gymnasium`, `stable-baselines3[extra]`, `tensorboard`, `imageio`. Actualicé `Scripts/requirements.txt`.
- Escribí `Scripts/rl_env.py` con la clase `DUMHeadTrackingEnv` heredando de `gymnasium.envs.mujoco.MujocoEnv` (v5+). Action space `Box(3,)` en [-1,1], observation space `Box(13,)`, frame_skip=4 → control a 50 Hz.
- Escribí `Scripts/smoke_test_env.py` para validar carga, espacios, reset, steps y trigger del hook de enfoque.
- Detecté en el smoke test que con σ=0.1 (propuesta original) `r_track ≈ 0` para cualquier θ > 11°, dejando al agente sin gradient útil. Subí σ a 0.4 (justificación numérica documentada en el código).
- Validé con un test temporal que el eje `HEAD_FORWARD_LOCAL=[0,-1,0]` coincide exactamente con la dirección "forward" del cuerpo `LenteExt_link` en pose neutra. Cada joint de la cabeza rota el forward de manera coherente (Neck pitch, HeadBase pitch, HeadRotation yaw).
- Escribí `Scripts/train_ppo.py`: entrenamiento PPO con SB3, soporta modo `--smoke` para validación rápida, paralelización vía `SubprocVecEnv`, checkpoints periódicos, tensorboard.
- Smoke training (20k steps, 1 env, 28 s wall-clock): pipeline OK, sin errores. `ep_rew_mean ≈ -35` esperable a esa cantidad de steps — PPO continuous control típicamente necesita 100k+ para que la curva empiece a subir.
- Lancé en background training real: 500k steps × 4 envs paralelos, hyperparams PPO estándar para MuJoCo continuous control (lr=3e-4, n_steps=2048, batch=64, n_epochs=10, gamma=0.99, gae_lambda=0.95). Esperado wall-clock ~4 minutos.
- Training completado en 7:11 wall-clock. `ep_rew_mean` subió monótonamente: -35 (inicio) → -17 (57k) → +173 (500k). `std` de la policy cayó a 0.42 indicando convergencia.
- Escribí `Scripts/eval_ppo.py` con evaluación determinística + generación de video MP4.
- Agregué `<visual><global offwidth="1280" offheight="720"/></visual>` al XML para soportar render offscreen (aunque MujocoEnv parece ignorarlo; el video final salió a 480×480 que es el framebuffer default).
- Instalé `imageio[ffmpeg]` para escritura MP4. Lo agregué a `requirements.txt`.
- Eval con 10 episodios determinísticos: `ep_rew_mean = +215`, 7/10 episodios alcanzan θ_min < 19°, 3/10 disparan el hook de enfoque.
- Generé `runs/ppo_dum_first/eval.mp4` (3 episodios, 30 s, 50 fps).
- **Diagnóstico de los 3 episodios fallidos**: descubrí que `HeadRotation_joint` (único joint que aporta yaw a la cabeza) tiene range ±0.186 rad ≈ ±10.7°, pero el sampling de target uso un cono de ±60° azimut. Cualquier target con azimut > 11° es físicamente imposible de alcanzar con solo los 3 joints del cuello (Neck y HeadBase son pitch, no yaw). La policy aprendió bien dentro de las restricciones físicas; la varianza alta no es un fallo de la policy sino una mismatch entre task y modelo.
- Decisión: expandir el range físico de `HeadRotation_joint` a ±60° (1.0472 rad) en lugar de achicar el cono de target.
  - Justificación técnica defendible: el servo MG90S elegido para ese joint rota 180° por especificación de fábrica. El ±10.7° previo es 17× más restrictivo que el motor real y probablemente derivó del mismo bug de conversión grados→radianes que afectó al joint range original (10.698 rad ≈ 612°, claramente grados sin convertir). ±60° es una restricción mecánica conservadora del diseño del lente, todavía dentro del rango físico del MG90S.
- También expandí `act_HeadRot ctrlrange` a ±1.0472 para coincidir con el nuevo rango del joint.
- Lancé en background training v2 desde cero (la policy v1 no es reusable porque el espacio de control cambió sustancialmente): 500k steps × 4 envs.

### TODO post v2: estabilizar el cuerpo (no la cabeza)

Observación del usuario sobre el video v1: el robot tiene un "vaivén" lateral permanente del torso. **Diagnóstico técnico:**
- Solo `Base_link` está anclado al suelo. Todos los demás cuerpos (cintura, hombros, brazos, muñecas, dedos) cuelgan en su cadena cinemática con `ctrl=0`.
- B1 del benchmark inicial lo confirmó: `|qvel|_max = 75 rad/s` con todos los actuadores en cero → algo está oscilando libre por gravedad.
- Como `LenteExt_link` (frame de la mirada) cuelga al final de la cadena `Base → BaseHip → FullBody → Neck → HeadBase → LenteExt`, **el agente recibe observaciones de cabeza desde una plataforma que se mueve sola**. Está aprendiendo a apuntar y a compensar oscilaciones simultáneamente.
- Efecto en RL: el gradient ve dos señales mezcladas (corrección lenta de orientación + compensación rápida del vaivén) y converge a una policy de precisión intermedia.

Fixes candidatos a probar después de v2:
1. Anclar el torso con position actuators de alto `kp` y setpoint=0 sobre `act_BaseHip` y `act_HipBody`. Esto los hace "rígidos" sin cambiar el action space del agente.
2. Aumentar `damping` de `BaseHip_joint` (actualmente 0) y `HipBody_joint` (35) para frenar la oscilación lenta.
3. Comandar pose neutra en TODOS los actuadores del cuerpo, no solo `ctrl=0`. Los brazos también cuelgan.

### Iteración v3 — Fix vaivén lateral + mejora visual de videos

- Identifiqué la causa exacta del vaivén lateral: `act_BaseHip` tenía `kp=1000, kv=0`. Sin componente derivativa, el actuador se comporta como un resorte sin amortiguador → el cuerpo entero oscila lateralmente cada vez que cualquier perturbación lo desplaza. El joint `BaseHip` es el único entre el suelo y el resto del robot, así que esa oscilación se propaga hacia arriba hasta la cabeza.
- Aplicado fix mínimo: `kv` de `act_BaseHip` de 0 a 80. No toqué nada más para mantener el cambio aislado y diagnosticable.
- Mejoré `Scripts/eval_ppo.py` para que los videos generados tengan overlay informativo en cada frame:
  - Esquina sup-izq: nombre del modelo, total_timesteps con que se entrenó, velocidad de reproducción (1× real-time @ 50 fps).
  - Esquina sup-der: episodio actual, step del episodio (X/500), θ en grados, ep_reward acumulado, estado del hook de enfoque.
  - El overlay usa PIL (`imageio[ffmpeg]` ya lo trae).
- Lancé training v3 en background: mismo setup que v2 (500k steps × 4 envs) sobre el modelo con `kv=80` en BaseHip. Para validar el fix se necesita re-entrenar porque la dinámica del modelo cambió.
- v3 terminó (7:30 wall-clock). `ep_rew_mean = 314` (vs 238 v2). Eval determinístico: `ep_rew_mean = 344 ± 109`, `θ_final = 3.99° ± 2.63°` (vs 15.8° en v2), focus disparado en 9/10. **El fix del vaivén multiplicó por 4 la precisión sostenida.**

### Iteración v4 — Bloqueo completo del BaseHip + nuevo training

- Decisión del usuario: en lugar de solo amortiguar con `kv=80`, **bloquear el `BaseHip_joint` completamente** con `range="-0.0 0.0"`. Argumento: la tarea de head-tracking no requiere yaw de cintura, eliminar el DoF elimina la fuente de oscilación de raíz.
- Defendible técnicamente: para una tarea específica que no usa el grado de libertad de cintura, congelarlo no compromete generalidad del robot. Para tareas futuras que sí lo usen, se reabre el range.
- Lanzado training v4 desde cero: 500k steps × 4 envs sobre el modelo con BaseHip bloqueado.
- v4 terminó (7:38 wall-clock). `ep_rew_mean = 300`. Eval determinístico: `ep_rew_mean = 323 ± 103`, `θ_final = 6.56° ± 2.05°`, focus 8/10.
- Resultado contraintuitivo: v4 es **ligeramente peor en métricas** que v3 (θ_final 6.5° vs 4°), pero **más consistente** (σ menor). Hipótesis técnica: el BaseHip con kv=80 (v3) actuaba como amortiguador con elasticidad residual que disipaba micro-momentos; al bloquearlo (v4), esos momentos se transmiten al HipBody (que está libre) y producen una oscilación pitch más alta.
- Visualmente, v4 muestra una cintura quieta sin vaivén lateral, lo que probablemente da mejor presentación visual aunque la precisión angular sea levemente inferior.

### Iteración v5 — Eliminación completa del joint BaseHip

- Decisión del usuario: en lugar de limitar el `BaseHip_joint` a un rango infinitesimal (v4, `±0.0001`), **comentar el joint completamente** del XML. Sin joint, `BaseHip_link` queda solidario rígido a `Base_link` — MuJoCo lo trata como un solo cuerpo a efectos dinámicos. También se comentó el actuador `act_BaseHip` que dependía del joint inexistente. `nu` cae de 14 a 13.
- Ventaja sobre v4: ningún DoF residual (eliminado el artefacto numérico del rango `±0.0001`), simulación más limpia, y reproducción del modelo en otros simuladores no depende de tolerancias.
- El env de RL no requirió cambios: los actuadores se resuelven por nombre (`mj_name2id`), no por índice.
- Lanzado training v5: 500k steps × 4 envs sobre el modelo con BaseHip eliminado como joint.
- v5 terminó (8:04 wall-clock). Resultado peor que v3/v4: `ep_rew_mean = 234` training, `255 ± 188` eval, `θ_final = 17.9° ± 16.2°`, focus 3/10. Eval episodio por episodio mostró polarización: 3 excelentes / 4 medios / 3 fallados. Diagnóstico: convergencia prematura — la policy colapsó con `std=0.243` y `entropy_loss=-0.022` (casi cero entropía) antes de explorar lo suficiente. Hipótesis técnica: cambio de matriz de inercia al fusionar Base_link+BaseHip_link en un cuerpo rígido cambió la dinámica de respuesta; combinado con un mal seed inicial, la policy quedó atrapada en local optimum.

### Iteración v6 — Clean slate con kp aumentado

- Decisión del usuario: empezar de cero. Eliminar todas las policies previas, subir el `kp` de los 3 head actuators para que el robot tenga respuesta más rápida sin penalizar el effort en el reward.
- Eliminé `runs/` completo.
- Cambios mínimos en el XML, sin tocar `forcerange` (sigue al torque máximo del servo real):
  - `act_Neck`: kp 1100 → 1500 (+36%)
  - `act_HeadBase`: kp 1000 → 1500 (+50%)
  - `act_HeadRot`: kp 1300 → 1800 (+38%)
- Argumento defendible: kp más alto sólo hace que el motor llegue a saturación de torque más rápido — el techo físico de fuerza (2.11 N·m del DS3218/MG90S) sigue siendo el mismo. Es equivalente a "afinar mejor la ganancia del controlador" sin cambiar el motor.
- Lanzado training v6: 500k steps × 4 envs sobre el modelo nuevo.
- v6 terminó. `ep_rew_mean = 236` train, `255 ± 190` eval, `θ_final = 18.5°`, focus 3/10. Patrón idéntico a v5: mismos 3 episodios fallidos, mismos resueltos. Subir kp solo no movió la aguja.

### Iteración v7 — ent_coef + rango ±90° + 1M steps

- Modificado `train_ppo.py`: parámetros `--ent-coef` y `--learning-rate` configurables, banner de inicio/fin con timestamps y duración total.
- Expandido `HeadRotation_joint` de ±60° a ±90° (1.5708 rad), defendible como rango físico real del servo MG90S.
- Lanzado v7 con `--ent-coef 0.005 --steps 1000000`.
- v7 terminó en 14:01 wall-clock. `ep_rew_mean = 212` train, `255 ± 188` eval, `θ_final = 17.8°`, focus 2/10. **Patrón idéntico a v5 y v6**. Confirmado: el problema es estructural del modelo sin BaseHip, no del algoritmo de aprendizaje.

### Iteración v8 — Consulta al experto + recalibración de PID + cambios al reward y obs

- Usuario reporta 3 problemas visuales en el video de v7: (a) llegada secuencial joint-por-joint, (b) movimiento lento, (c) vibración al asentarse.
- Consulta delegada a un agente experto en MuJoCo y control. Diagnóstico cuantitativo del experto:
  - **Joint por joint**: ζ_Neck = 3.6, ζ_HeadBase = 4.8, ζ_HeadRot = 0.72 (calculados con J_efectivo). El HeadRot llega rápido, los pitch arrastran porque están fuertemente sobreamortiguados.
  - **Vibración**: no viene de los head joints (sobreamortiguados); viene de `HipBody_joint` con ζ ≈ 0.47 (subamortiguado). El contragolpe de la cabeza llegando excita al HipBody que se balancea y propaga.
  - **Damping del joint se sumaba al kv del actuator** → era amortiguamiento "oculto" que nadie había considerado. Total efectivo de Neck antes: kv + damping = 60 + 35 = 95.
- Plan de recalibración aplicado al XML:
  - `act_Neck`: kp 1500 → 3000, kv 60 → 24
  - `act_HeadBase`: kp 1500 → 3000, kv 80 → 24
  - `act_HeadRot`: kp 1800 → 3000, kv 20 → 38
  - `act_HipBody`: kp 1000 → 4000, kv 20 → 90 (sobreamortiguar a propósito)
  - `Neck_joint`, `HeadBase_joint`, `HeadRotation_joint`: damping 35 → 5
  - `HipBody_joint`: damping 35 → 50
- Cambios en `rl_env.py`:
  - `w_smooth`: 0.05 → 0.15 (comandos más suaves contra chattering)
  - `w_jitter`: 0.0 → 0.002 (penaliza ||qvel_head||² para que la policy aprenda a llegar con velocidad baja)
  - Sumado `qpos` y `qvel` de `HipBody_joint` al observation space (shape 13 → 15). El agente ahora "ve" al torso oscilar y puede compensar activamente.
- Lanzado training v8 con `--ent-coef 0.005 --steps 1000000`. Duración: 13:51 wall-clock.
- v8 eval: `ep_rew_mean = 307 ± 189`, `θ_final = 14.4°`, focus 4/10. Mejor que v7 en todas las métricas (+20% reward, −19% θ_final, doble focus). 5/10 episodios con `θ_final < 5°` (precisión excelente). Los mismos 2 seeds (ep 3, 4) siguen siendo estructuralmente difíciles.

### Iteración v9 — Feedback visual del usuario sobre v8

- Usuario reporta sobre v8: (a) joints llegan más juntos ✓ y se mueve más rápido ✓; (b) sigue vibrando al asentarse; (c) en algún episodio no gira al target; (d) los dedos se ven moverse en el video.
- Aclarado al usuario: los dedos NO afectan al training (el agente ni los observa ni los controla). Se mueven por física pura. `act_LeftShoulderArm` con kv=0 es el origen pero es solo cosmético.
- Pedido adicional del usuario: que la animación de enfoque sea más orgánica (1-3 ciclos random, profundidad random, resting position random).
- Pedido adicional: videos llamados `eval_v#.mp4` con # extraído del run name.
- Cambios aplicados:
  - `w_jitter`: 0.002 → 0.01 (5× más penalización por velocidad al asentar; debería reducir la vibración).
  - `ent_coef`: 0.005 → 0.01 (más exploración para los seeds difíciles).
  - Hook de enfoque randomizado en cada reset:
    - `n_cycles ∈ {1, 2, 3}` uniformes.
    - `amplitude ∈ [0.5, 1.0]` del lens_max.
    - `resting offset ∈ [0, 0.15]` del lens_max (queda ligeramente extendido al final, no en 0 exacto).
  - `eval_ppo.py`: `--video` sin path autogenera `eval_v<N>.mp4` extrayendo N del run name.
- v9 terminó en 13:46 wall-clock. Eval: `ep_rew_mean = 301 ± 192`, `θ_final = 14.8°`, focus 5/10. Métricas prácticamente iguales a v8 (era esperable: las modificaciones apuntaban a calidad visual de la policy, no a alcance numérico). Mismos 2 seeds siguen fallando.

### Iteración v10 — Validación del axis + reward de focus + cono ampliado + actuadores faltantes

- Feedback del usuario sobre v9: (a) el focus se ve más orgánico (random funciona), (b) en ep 2 el vector perpendicular al lente parece quedar lejos del target rojo (sospecha de mismatch en HEAD_FORWARD_LOCAL), (c) sigue temblando, (d) querer reward extra por hacer y mantener focus, (e) jitter general del modelo al cargar.
- Cambios en el XML:
  - Agregado site `aim_indicator` verde a 10 cm en `-Y` local del `LenteExt_link`. Valida visualmente que la dirección "forward" del código coincide con el rendering.
  - `HeadRotation_joint` y `act_HeadRot` range: ±90° → ±105° (1.8326 rad). Más margen de yaw.
  - Damping 0.75 + frictionloss 0.01 agregado a los joints sueltos del brazo izquierdo (`LeftForearm_joint`, `LeftWrist_joint`, `LeftLever_Slider`). Matchea damping del lado derecho. Reduce jitter pasivo al cargar el modelo.
  - `act_LeftShoulderArm` kv: 0 → 20 (era el único sin amortiguamiento del agente, oscilación visible).
  - Actuadores nuevos: `act_BodyShoulderLeft` y `act_BodyShoulderRight` (kp=800, kv=30, forcerange=±1.08 N·m, range ±5°). Habían quedado como joints sin actuador. `nu` pasó de 13 a 15.
- Cambios en `rl_env.py`:
  - `TARGET_CONE_AZIMUTH`: 60° → 75°. `TARGET_CONE_ELEVATION`: 30° → 45°. Más exigente, fuerza a la policy a usar el rango completo del cuello.
  - Nuevos componentes del reward:
    - `w_focus_event = 5.0`: bonus puntual al disparar focus por primera vez en el episodio.
    - `w_focus_hold = 1.0`: bonus por step mientras `focus_triggered AND θ < 0.07 rad`. Premia mantener.
    - `w_post_focus_mult = 2.0`: multiplica `w_smooth` y `w_jitter` por 2 cuando `focus_triggered`. Refuerza la quietud post-focus.
- Cambios en `eval_ppo.py`: default de episodios 10 → 15.
- v10 terminó en 13:27 wall-clock. `ep_rew_mean = 318` train (subiendo monótonamente cuando terminó). Eval 15 ep: `ep_rew_mean = 303 ± 351`, `θ_final = 23.4°`, focus 4/15. Récord individual: ep 6 con reward **+972** (focus YES, θ_final=0.7°). Los rewards de focus_event y focus_hold acumulados dan hasta +500 extra cuando la policy mantiene el target. Proporción de episodios fallidos subió a 33% (vs 20% en v9) por el cono ampliado más exigente.

### Sprint anti-jitter (no es una versión nueva del PPO sino estabilización del modelo físico)

- Feedback del usuario: al cargar el modelo en MuJoCo, los joints cambian de posición de manera errática permanente. Quiere que se queden quietos a menos que un actuador los mueva. Pedido explícito: validar la solución con el experto.
- Creé `Scripts/_jitter_diag.py` (temporal): simula 5s con ctrl=0 en todos los actuadores y mide `qvel_rms`, `qvel_max`, `qpos_drift` por joint. Threshold de jitter = `qvel_rms > 0.05 rad/s`.
- Estado inicial: **11 joints** sobre threshold. `||qvel||_max global = 2.91 rad/s`, `||qvel||_rms = 0.22`. Los peores: `LeftFingerTop` (qvel_max 2.56), `LeftFingerBot` (qvel_max 2.91).
- Origen identificado iterativamente con datos:
  1. **Asimetría XML lado derecho**: `act_RightLever_Slider` con `kv=0` + `RightLever_Slider` sin damping. Equality `polycoef=34.9` amplificaba a los dedos ×35.
  2. **Finger joints sin damping** (solo frictionloss=0.01).
  3. **kp sobredimensionado** en torso y cabeza, causando ω_n cerca del límite de resolución del integrador.
- Fixes aplicados iterativamente, midiendo después de cada uno:
  1. Igualé `act_RightLever_Slider` al izquierdo (kv=0→20, kp=1000→1400) y agregué damping=0.75 + frictionloss=0.01 al joint.
  2. Damping=0.5 + frictionloss=0.01 en los 4 finger joints.
  3. `LenteExt_joint` damping 0.75→5.
  4. `HipBody_joint` damping 50→150.
  5. Joints de hombros (4 joints) damping 1.0→5.0.
- Después de los fixes de damping, 3 joints quedaban en jitter (HipBody, HeadBase, RightShoulderArm). Consulté al experto MuJoCo con datos en mano. **Diagnóstico clave del experto**: el problema NO era falta de damping, era **kp sobredimensionado** que ponía ω_n cerca del límite de muestreo del integrador (11 muestras/período con dt=0.005). Plan: bajar kp drásticamente.
- Plan del experto aplicado tal cual:
  - `act_HipBody`: kp 4000→800, kv 90→40
  - `act_HeadBase`: kp 3000→600, kv 24→18
  - `act_LeftShoulderArm`: kp 900→500, kv 20→16
  - `act_RightShoulderArm`: kp 1400→500, kv 20→16
- Estado final: **cero joints sobre threshold**. `||qvel||_max global = 0.115 rad/s` (24× mejor), `||qvel||_rms = 0.021` (10× mejor). Validación a 30s confirma que no hay drift acumulativo en los joints principales. Drift residual de ~2° en fingers explicado por la equality polycoef soft (no es bug).
- Veredicto final del experto: **"Aceptable, listo para re-entrenar."** Offset estático de 1.8° en HipBody es la cesión real de un DS3218 bajo carga, defendible.
- Borré el script `_jitter_diag.py` (servirá la próxima vez, fácil de recrear si hace falta).

### Iteración v11 — Target dinámico (estático + lineal + circular)

- Implementé 3 modos de movimiento del target en cada episodio (uno random):
  - `static`: target fijo en un punto sampleado del cono (comportamiento previo).
  - `linear`: oscila entre 2 puntos A↔B con período 4-8s. Sample cosenoidal.
  - `circular`: orbita alrededor de un punto del cono, radio 5-15 cm, omega 0.5-1.5 rad/s (~30-90°/s). Plano perpendicular a la dirección desde la cabeza.
- En cada step se actualiza `self._target_world` según el modo y `sim_time`, y se setea `data.mocap_pos` para que la esfera roja siga la trayectoria.
- `info` del step incluye `target_mode` para análisis post-eval.
- Overlay del video en `eval_ppo.py` muestra `target: <mode>` para que se vea en pantalla qué tarea está enfrentando el agente.
- Lanzado v11 con 1.5M steps × 4 envs (más que v10 porque la tarea es más difícil). Duración 21:11 wall-clock.
- v11 eval (15 ep): `ep_rew_mean = 282 ± 273`, `θ_final = 23°`, **focus 8/15 (53%)** — el doble que v10 con tarea más difícil. Mejor episodio: ep 11 con θ_min=0.06°, θ_final=1.0°, reward 936 (focus YES). El alto θ_final en muchos episodios es engañoso: el agente alcanza el target (θ_min muy bajo, ej 0.06°) pero después el target se mueve y queda persiguiéndolo.

### Iteración v12 — Modelo limpio (jitter + axis + HipBody + focus dinámico) + reentrenamiento

- Tras el sprint anti-jitter + los fixes del experto (HeadRot exclude colisión, HipBody damping=40 + gear=3, axis bolita verde, focus indicator dinámico), reentrené desde cero.
- v12 setup: 1.5M steps × 4 envs, ent_coef=0.01.
- v12 eval (15 ep): `ep_rew_mean = 705 ± 305`, `θ_final = 11°`, **focus 14/15 (93%)**. 9/15 episodios con reward > 800. Mejor: ep 11 con reward 976.
- Salto cualitativo: x2.5 en reward respecto a v11 gracias a que el modelo físico estable permitió al agente entrenar sin compensar ruido espurio.

### Iteración v13 — Optimización de hardware + reward sqrt(streak) + site origen + obs con target_vel

- Consulta al experto sobre uso del hardware: con 16 logical CPUs (~8 físicos) usábamos solo 4 envs. Receta del experto:
  - `--n-envs 8` (1 por core físico, sweet spot).
  - `n_steps 2048 → 1024` (mantener rollout total).
  - `batch_size 64 → 256` (mejor uso de BLAS).
  - `net_arch [64,64] → [128,128]` (reward multi-componente justifica más capacidad).
  - `torch.set_num_threads(2)` para no pelear con MuJoCo workers.
- Modificado `train_ppo.py` con args para `--n-steps`, `--batch-size`, `--n-epochs`, `--net-arch` configurables.
- Feedback del usuario sobre v12: (1) delay del lente respecto al target en movimiento, (2) desfase constante entre dirección del lente y target, (3) querer reward creciente por mantener focus (no por iniciarlo). Pedido: validar con el experto.
- Consulta al experto:
  - **Desfase constante**: dominante hipótesis (3) — el origen del cálculo de θ (`body.xpos` del LenteExt_link) está a 0.37 m del centro visual del lente. La policy minimiza ángulo desde el origen del body, no desde el lente real → sesgo geométrico fijo. Fix: usar un `site` posicionado en el centro visual y leer `site_xpos`.
  - **Delay**: agregar `target_vel_local` a la observación para que la policy anticipe. NO bajar frame_skip (duplica costo).
  - **Reward de hold**: cambiar de constante a `w_hold · sqrt(streak_steps)` con `w_hold = 0.03`. Sublineal premia mantener prolongado sin explotar. Eliminar `w_focus_event` (escalón discreto que confunde value function).
- Aplicado:
  - XML: site `lens_center` en `pos="-0.274 0.013 0.220"` (CoM del LenteExt) dentro del body.
  - `rl_env.py`: cálculo de θ usa `data.site_xpos[lens_center_id]` como origen. `_target_local` y `_angle_to_target` reescritos.
  - Obs Box(15) → Box(18) agregando `target_vel_local` (diferencia finita target_world entre steps, proyectada al frame del lente).
  - Reward: `w_focus_event = 0.0`, `w_focus_hold = 0.03` con `sqrt(streak)`, `w_post_focus_mult = 1.5` (bajado de 2.0). Streak resetea cuando θ sale del umbral → no se puede farmear "entrar y salir".
- v13 setup: 3M steps × 8 envs, n_steps=1024, batch=256, net_arch=[128,128], ent_coef=0.01. FPS observado: **3395 vs 1200 de v12** (2.8× más rápido).
- v13 wall-clock: 20:06 — prácticamente igual a v12 a pesar de 2× los steps.
- v13 eval (15 ep): `θ_final = 4.9° ± 5.1°` (vs 11° de v12), **focus 15/15 (100%)**, **std del reward 92 (vs 305 de v12, 3.3× menos varianza)**. Cero episodios fallados (peor: 15.3°). En v12 había 2 con θ_final > 19°. θ_min < 1° en TODOS los 15 episodios. El reward total no es comparable directamente porque cambió la fórmula (sin focus_event, con sqrt en hold).

---

## Borrador del capítulo RL para el .docx (extraer lo que sirva)

> Material para que Elias seleccione/edite/descarte. Está armado en bloques que mapean a secciones típicas de un documento de proyecto final.

### Sección propuesta: "Control aprendido mediante RL"

#### 1. Motivación y alcance

El control clásico basado en interpolación de keyframes funciona para animaciones predeterminadas pero falla cuando el robot debe reaccionar a estímulos externos en tiempo real. Para demostrar la viabilidad de un robot animatrónico **reactivo**, se implementó una capa de control aprendido mediante Reinforcement Learning (RL) que orienta la cabeza del Pit Droid hacia un objetivo visual móvil.

Esta capa complementa (no reemplaza) al modelo físico simulado: el agente aprende qué comandos enviar a los actuadores que ya fueron calibrados en la sección anterior, transformando el robot de un actor "guionado" a un sistema que percibe y responde.

#### 2. Marco teórico

- **Reinforcement Learning**: paradigma donde un agente aprende una *policy* π(a|s) — una función que mapea estados a acciones — mediante interacción con un *environment* que devuelve recompensas. El agente busca maximizar la suma esperada de recompensas futuras.
- **Proximal Policy Optimization (PPO)**: algoritmo *on-policy* de gradiente de política, propuesto por OpenAI en 2017. Estándar de facto para control continuo (locomoción, manipulación). Estable, sample-efficient y con pocos hiperparámetros críticos.
- **Gymnasium**: librería estándar para definir environments de RL (fork mantenido de OpenAI Gym). Su clase `MujocoEnv` integra directamente con las bindings nuevas de MuJoCo (no las deprecadas de `mujoco-py`).
- **Stable-Baselines3 (SB3)**: implementación de referencia en PyTorch de PPO, SAC, TD3 y otros. Usada en este proyecto por su madurez y reproducibilidad.

#### 3. Diseño del environment

El environment `DUMHeadTrackingEnv` hereda de `gymnasium.envs.mujoco.MujocoEnv` y expone la tarea como un Markov Decision Process.

**Action space**: `Box(3,)` en el rango `[-1, 1]`. Cada dimensión corresponde a un comando normalizado para uno de los tres actuadores que controlan la cabeza:
- `act_Neck` (pitch del cuello)
- `act_HeadBase` (pitch corto de la cabeza)
- `act_HeadRot` (yaw, rotación lateral)

La normalización a `[-1, 1]` es práctica estándar: facilita el aprendizaje al evitar magnitudes asimétricas. El env mapea internamente cada valor al `ctrlrange` correspondiente del actuador.

**Observation space**: `Box(18,)`, compuesta por:

| Dimensiones | Variable | Por qué |
|---|---|---|
| 3 | `qpos` de los joints del cuello | dónde tiene la cabeza el robot ahora |
| 3 | `qvel` de los joints del cuello | a qué velocidad va — anticipar frenado |
| 1 | `qpos` del HipBody | "ve" el torso oscilando (puede compensarlo) |
| 1 | `qvel` del HipBody | velocidad angular del torso |
| 3 | Vector target→cabeza (frame local del lente) | dónde está el target relativo al lente |
| 1 | Distancia escalar al target | redundante pero ayuda |
| 3 | Velocidad del target (frame local) | permite anticipar movimiento dinámico |
| 3 | Acción previa `a_{t−1}` | necesaria para penalizar cambios bruscos |

**Reward function**: la señal de aprendizaje. Se diseñó iterativamente y la versión final es:

```
r(s, a) = w_track · exp(−θ² / σ²)               (1) tracking principal
         − w_smooth · m · ‖a_t − a_{t−1}‖²       (2) penalización a cambios bruscos
         − w_effort · ‖a_t‖²                     (3) penalización al esfuerzo
         − w_jitter · m · ‖q̇_head‖²              (4) penalización a vibración
         + w_hold · √(streak)   si θ < umbral    (5) bonus por mantener target
```

donde `θ` es el ángulo entre el vector "mirada" del lente y el vector cabeza→target; `m` es un multiplicador (1.5) que se activa cuando el agente ha logrado focus al menos una vez en el episodio (refuerza la quietud post-éxito); y `streak` cuenta los pasos consecutivos con θ < umbral.

Valores finales: `w_track=1.0`, `σ=0.4 rad`, `w_smooth=0.15`, `w_effort=0.0075`, `w_jitter=0.01`, `w_hold=0.03`, `umbral=0.07 rad (≈4°)`.

**Justificaciones de diseño**:
- *exp en (1) vs cuadrática*: la cuadrática castiga muy fuerte a distancias grandes, generando una policy de "pánico". El exponencial satura suavemente; el agente puede explorar sin recibir señales tóxicas.
- *√(streak) en (5)*: lineal sin cota desestabiliza el gradiente; lineal con cap es discontinua; sqrt premia más los primeros steps de focus (los más costosos) y satura naturalmente.
- *Multiplicador post-focus en (2) y (4)*: la cabeza animatrónica debe quedarse quieta después de mirar — más penalty a la velocidad y al chattering cuando ya alcanzó el target.

**Modos de target**: en cada episodio se sortea uniformemente uno de tres modos:
- `static`: target fijo en un punto del cono frontal (±75° azimut × ±45° elevación).
- `linear`: oscila entre dos puntos del cono con período 4-8 s.
- `circular`: orbita alrededor de un punto a 5-15 cm de radio, con velocidad angular 30-90 °/s.

Esta diversidad fuerza al agente a aprender estrategias robustas tanto para fijación como para seguimiento.

**Episodios**: 500 pasos de duración (10 s a 50 Hz de control). Sin terminación temprana: la tarea es continua, el agente persigue al target indefinidamente.

#### 4. Pipeline de simulación

- Timestep del solver: 5 ms con integrador `implicit` (compromiso entre estabilidad y precisión).
- Frame skip: 4 pasos del solver por step del agente → control efectivo a 50 Hz.
- 8 environments paralelos vía `SubprocVecEnv` (1 por core físico del CPU).
- Throughput observado: 3.395 steps/segundo, 3M steps en ~20 min wall-clock.

#### 5. Sprint iterativo de entrenamiento

El proceso fue inherentemente iterativo: cada training revelaba comportamientos no anticipados que motivaban refinamientos del modelo físico o del reward. La tabla siguiente resume las 13 versiones entrenadas:

| Versión | Cambio principal | Eval θ_final | Focus rate | Lección |
|---|---|---|---|---|
| v1-v3 | Bugs XML, integrator, validación de axis | — | — | Setup base; identificación temprana de bugs (ctrlrange, rangos en grados sin convertir) |
| v3 | BaseHip con kv=80 | 4.0° | 9/10 | Primer baseline funcional |
| v4-v7 | Pruebas sin BaseHip; ent_coef, kp variados | 14-18° | 3-4/10 | Sin BaseHip, PPO converge a local optimum; el DoF residual era un estabilizador dinámico |
| v8 | PID recalibrado por experto (kp drásticamente más bajos) | 6.6° | 8/10 | kp sobredimensionado ponía ω_n cerca del límite del integrador; kp = 17× lo necesario causaba jitter numérico |
| v9-v10 | Random focus animation, reward de focus, cono ampliado | 14-23° | 4-8/15 | Cono más exigente revela las limitaciones reales |
| v11 | Target dinámico (3 modos) | 23° | 8/15 | Doble focus rate vs estático aunque tarea más difícil |
| v12 | Modelo limpio (jitter eliminado, axis correcto, focus dinámico) | 11° | 14/15 | Salto cualitativo: el jitter espurio del modelo físico mascaraba el aprendizaje |
| **v13** | **Site para origen θ + target_vel + sqrt(streak), 8 envs, red [128,128]** | **4.9° ± 5.1°** | **15/15** | **Convergencia, precisión sub-grado, mínima varianza** |

#### 6. Sprint anti-jitter (caso de estudio)

Al cargar el modelo en MuJoCo, los joints sin damping adecuado oscilaban permanentemente por gravedad y por restricciones de equality con coeficiente alto. Medición inicial: `||qvel||_max = 2.91 rad/s`, 11 de 19 joints sobre el threshold. Los peores eran los dedos (`LeftFingerTop/Bot` con qvel_max 2.5-2.9 rad/s), causa: equality polycoef ×34.9 amplificaba cualquier oscilación del slider.

Iteración con consulta a un experto externo en MuJoCo:
1. Igualar damping de joints (asimetría izq/der descubierta en `act_RightLever_Slider kv=0`).
2. Agregar damping a finger joints y al LenteExt.
3. Diagnóstico clave del experto: **kp sobredimensionado** ponía la frecuencia natural cerca del límite de resolución temporal del integrador (11 muestras por período con dt=0.005). Bajar kp 17× resolvió el jitter sin agregar damping pasivo.

Resultado final: `||qvel||_max = 0.115 rad/s` (24× mejor), **cero joints** sobre threshold.

#### 7. Hardware y reproducibilidad

- CPU: 16 logical processors (~8 físicos). Sin GPU dedicada.
- Stack: Python 3.11, MuJoCo 3.8, PyTorch (CPU), Stable-Baselines3, Gymnasium.
- Tiempo por iteración de entrenamiento (3M steps × 8 envs): 20 min wall-clock.
- Evaluación determinística de 15 episodios + generación de video con overlay: 1 min adicional.
- `Scripts/benchmarks.py` permite re-validar las características físicas del modelo (estabilidad, throughput, convergencia de actuadores) en cualquier momento.

#### 8. Resultados finales

La policy `ppo_dum_v13` alcanza precisión sub-grado en el 100% de los episodios evaluados sobre 15 muestras con seeds distintos y modo de target aleatorio. El `θ_final` mediano es 4.9° con desviación 5.1°. En episodios donde el target es estático o de movimiento lento, el robot mantiene el target dentro de ±1° por más de 8 segundos consecutivos.

El comportamiento del enfoque animatrónico (oscilación del lente interno) se dispara automáticamente la primera vez que la cabeza alcanza al target, con parámetros aleatorizados por episodio (1-3 ciclos, profundidad 50-100% del rango del slide, posición final variable), reproduciendo el carácter "vivo" característico de un robot animatrónico profesional.

#### 9. Conclusiones técnicas

- El reward shaping es la actividad más delicada y de mayor impacto en este tipo de tarea. Tres iteraciones críticas del reward (σ del exponencial, `w_focus_hold` con `sqrt(streak)`, multiplicador post-focus) cambiaron drásticamente el comportamiento aprendido sin alterar el algoritmo ni los hiperparámetros.
- El modelo físico **debe** estabilizarse antes de entrenar. Un robot que tiembla al cargar nunca entregará una policy estable, por más sofisticada que sea la red.
- La validación visual del axis del lente expuso una clase de bug que las métricas numéricas no detectaron: el reward optimizaba un ángulo medido desde un origen geométrico incorrecto (offset de 0.37 m entre el body origin y el centro visual del lente). El uso de un site dedicado (`lens_center`) resolvió el "desfase constante" reportado.
- La velocidad del target en la observación es barata (3 dims) y de altísimo ROI: permite que la policy *anticipe* el movimiento en vez de *reaccionar* tardíamente.

#### 10. Próximos pasos

- Extender la policy a tareas múltiples coordinadas (cabeza siguiendo target + brazo saludando + pinza agarrando).
- Implementar interfaz de control remoto para que un operador pueda comandar el target desde teclado/joystick.
- A largo plazo: transfer learning de la policy simulada al robot físico impreso en 3D (sim-to-real), incorporando *domain randomization* sobre masas, fricción y latencias.

---

(Fin del borrador del capítulo)
Training (cambiá <nombre> por la versión que quieras):

  # (correr desde la raiz del repo)
  python Scripts/train_ppo.py --steps 500000 --n-envs 4 --name <nombre>

  Eval + video (mismo <nombre>):

  python Scripts/eval_ppo.py runs/<nombre>/final.zip --episodes 10 --video runs/<nombre>/eval.mp4

  Opciones útiles del train_ppo.py:
  - --steps 1000000 para entrenar más tiempo (~14 min con 4 envs).
  - --n-envs 8 si querés más paralelización (los 8 envs paralelos andan ~1.5× más rápido que 4).
  - --smoke para validar pipeline en 20k steps (~30 s).
  - --resume runs/<nombre>/checkpoint_XXX_steps.zip para continuar desde checkpoint.

  Opciones útiles del eval_ppo.py:
  - --episodes N cambia cuántos episodios evalúa.
  - Sin --video solo imprime stats, no graba.
  - --stochastic usa la política con ruido (en vez de determinístico) — útil para ver variabilidad.
---


# History — Sprint Grab (RL "agarrar y lanzar")

Tarea episódica: una bola amarilla cae desde encima de la palma del robot; éste debe interceptarla con la pinza, sostenerla un rato y luego lanzarla hacia adelante. Lado del brazo (R o L) se randomiza por episodio. Compartimos motor de simulación y mismo modelo MJCF con el head-tracking previo (`DUM4_grab.xml` agrega bola + sites + `<connect>` equality).

## 2026-05-17 — v1: bootstrap

- Cloné `DUM4.xml` → `DUM4_grab.xml`. Agregué `<body name="yellow_ball">` con `<freejoint>`, bola radio 4 cm, masa 50 g; sites visuales `grip_R` / `grip_L` (esferas verdes 1.2 cm) que marcan el centro de la palma; equality `<connect>` que une rigidamente cada wrist con la bola, inicialmente desactivada.
- Implementé `Scripts/rl/envs/grab_env.py` (Gymnasium `MujocoEnv`, obs Box(24,), action Box(5,)). State machine `FALLING → HELD → THROWN`, activación/desactivación del connect según detección de grab/release, patch de `qvel` de la bola al soltarla para que conserve velocidad lineal del EE.
- Reward shaping de plan: `r_approach` (negativo, premia proximidad EE-bola), `r_grab_first=500` one-shot, `r_grab_height`, `r_grab_speed`, `r_hold` piecewise, `r_throw` por distancia (tanh saturado), `r_floor_fail` proporcional a la fase, penaltys de smoothness e indecisión.
- **Bug crítico**: sites `grip_R/L` aparecían MUY lejos de los dedos en el render (offset de ~30 cm). Causa: el exportador ACDC4Robot de Fusion360 escribe las mallas STL en coordenadas del *origen del ensamble*, no del body, así que `body_xpos` ≠ posición visual del mesh. Lo recomendó el experto MuJoCo: usar **promedio de `xipos` (CoM en world)** entre los bodies `*TopFinger_link` y `*BotFinger_link`, expresado en frame local del wrist. Fixed.

## 2026-05-17 — v2: long training con bola gigante

- Primer training largo: 5M steps, 16 envs, ~31 min. ep_rew_mean training pasó de -714 → -437. Eval (12 episodios, deterministic): **0/12 grabs**, ep_rew_mean = -196 ± 193, std final de la policy = 1.26 (todavía exploración alta).
- Diagnóstico: la policy explora mucho en training (consigue grabs ocasionales que mantienen la métrica) pero en eval determinístico colapsa a una pose que evita penaltys sin intentar el grab.

## 2026-05-18 — v3: bola chica (smallball)

- Reducí bola a radio **1.6 cm** (40 % del original) y masa **10 g** (PLA con infill 20-30 %, validado por experto). Bajé `GRAB_DIST_THR` de 0.06 → 0.03 m porque con bola chica el contacto físico es a 2.8 cm centro-centro.
- Amplié rango de los forearms: de [-0.349, +0.698] rad (-20°/+40°) a ±1.5708 rad (±90°) tanto en `joint range` como en `actuator ctrlrange`. Experto confirmó que `forcerange=±2.11 N·m` sigue teniendo margen 4-10× sobre el peor caso estático.
- Training: 3M steps, 16 envs, 19 min. ep_rew_mean -87 (train), **-71.6 ± 2.35 (eval)**, std=0.487, explained_variance=0.988. Eval: **0/12 grabs** otra vez, pero la policy convergió a una "no-acción" estable.

## 2026-05-18 — v4: workspace ampliado + bola más lenta (ws)

- Usuario amplió manualmente los hombros: `RightShoulderArm_joint` y `LeftShoulderArm_joint` ahora ±0.7418 rad (±42.5°) — antes ±5°/±15°. Esto da la palma un alcance frontal de ~0.45 m (vs ~0.36 m antes).
- Apliqué tres cambios pedidos por el usuario y validados con experto:
  1. Gravedad de Fase 1 reducida a **-6.0 m/s²** (caída ~26 % más lenta). Fase 2/3 vuelven a -9.81.
  2. Spawn de la bola: variación 1D sobre eje Y world ±0.13 m (antes era jitter circular ±5 mm). Centro corrido a `(-0.012, -0.275)` para R, mirror para L — fuerza a la policy a extender o retraer el brazo según donde cae.
  3. Confirmación: no tocar `forcerange` (peor caso estático ≈ 0.46 N·m vs 2.11 N·m disponible).
- Training: 3M steps, 16 envs, 27 min. ep_rew_mean -108 (train), **-110 ± 14 (eval)**, std=0.562, explained_variance=0.997. **0/12 grabs**.
- Diagnóstico: la policy converge consistentemente a "quedarse quieta" porque el `r_approach` negativo, los penaltys de effort/smooth y el bonus `r_grab_first=500` (sparse, lejano) la atrapan en un mínimo local. El "stuck-at-do-nothing" se mantiene a través de las tres iteraciones, indicando que el problema está en el shaping del reward, no en el espacio de acción ni la dinámica.

## 2026-05-18 — v5: rediseño del reward (en curso)

Usuario rediseñó la función de recompensa por completo para destrabar el reward sparsity:

- **`r_approach` ahora POSITIVO**, lineal saturado entre 200 pts/s (dist=0) y 15 pts/s (dist ≥ 0.5 m). Acumula por step naturalmente (scale por dt=0.02 s). Cambia el "baseline" de la policy de "negativo creciente" a "siempre algo positivo si está cerca", forzándola a aproximarse en vez de retirarse para evitar costos.
- **Detector de grab más estricto**: además del `dist < 0.025 m` actual, ahora pide `|v_ball - v_ee| < 0.5 m/s` (velocidades coincidiendo) y `|z_ball - z_ee| < 0.015 m` (alturas alineadas). Premio one-shot de +600 (era +500), conservando los bonos por altura del grab y por velocidad.
- **`r_floor_fail = -800` terminal en TODAS las fases** (antes era 0 en Fase 1). Acelera el descarte de policies pasivas.
- **Detector de throw refinado**: el evento "lanzamiento" sólo cuenta si al abrir los dedos hay aceleración lineal del EE > 8 m/s² proyectada radialmente hacia afuera del torso. Computada via `mj_objectAcceleration` (world frame) cacheada del step previo al release (porque al desactivar el connect el `qacc` se recompone). Si la policy "suelta" la bola sin imprimirle aceleración (drop), no obtiene recompensas de throw.
- **Recompensa de throw acumulativa por step** (antes era one-shot al touchdown): +4 pts por cada cm de distancia EE-bola por step, +2 pts por cm de altura por step, **-4 pts por cm de offset hacia atrás** del torso por step (penalty si tira hacia atrás). "Adelante" = -Y world (definido por la geometría de la palma extendida).
- Pendiente: ajustar `r_hold` (queda igual según spec del usuario) y revisar `r_indecision` para que conviva con el approach positivo.

Capítulo del experimento v5 a completarse cuando termine el training siguiente.

## 2026-05-19 — v5d / v5c / v5b: el espiral del reward sobre-diseñado

Después de v5 (rediseño completo del reward por spec del usuario), tres iteraciones fallidas en cadena:

- **v5b** (`P_FAIL_PHASE1=250` para suavizar el bootstrap): 0/12 grabs, ep_rew=-215 ± 20. La policy aprendió que el costo del piso supera al bonus de aproximación y se queda quieta minimizando esfuerzo. ep_len=31 (cae solita).
- **v5c_curr** (curriculum 1a static → 1b slow → 1c normal con callback de SB3 que dispara `set_subphase()` por `env_method`): la policy logró ep_rew ≈ +665 en sub-fase 1a (estática). Verificado por eval que **NUNCA disparaba el detector de grab** (`ep_len=251`, episodios completos hasta truncation). El "+665" venía exclusivamente de acumular el `r_approach` positivo hoverando cerca de la bola estática. Al transicionar a 1b/1c, el approach acumulado no compensaba el `r_floor_fail`, la policy regresaba a "no hacer nada".
- **v5d_1a** (mismo paradigma pero `r_approach` × 10 menor + decay temporal `(1 - t/T)` para anti-hover): ep_rew=+24.39 (eval, 1a estática). 0/12 grabs. El cap menor previno el hover-exploit, pero la policy tampoco descubrió el grab — quedó atrapada en un mínimo local de "movimiento mínimo".

**Diagnóstico estructural**: las 4 condiciones simultáneas del detector de grab v5 (`dist<0.025 + |v_ball-v_ee|<0.5 + |z_ball-z_ee|<0.015 + lever>0.007`) son individualmente plausibles por exploración random, pero **simultáneamente improbables**. Sumado al shaping multi-componente (approach con decay, indecision penalty, throw detector con `mj_objectAcceleration`, bonos múltiples al grab por altura/velocidad, floor_fail proporcional a fase), el espacio de reward es demasiado denso de discontinuidades y la señal del grab nunca emerge.

5 iteraciones, ~5h de cómputo acumulado, 0 grabs en eval. Conclusión: el reward complejo era el problema, no su calibración. Hay que simplificar.

## 2026-05-20 — v6: reset radical, reward minimalista

**Cambio de paradigma**: en vez de seguir ajustando capas, eliminar todas. Volver al "Sutton-style" reward shaping clásico (distance dense + sparse success bonus + small effort penalty) que funciona en MetaWorld, Isaac Gym, dm_control.

**Reward v6** (`Scripts/rl/envs/grab_env.py`):

```
r_step = -2.0 · dist(EE, ball)        # dense, negativo, continuo
       + -1e-4 · Σ τ²                 # effort penalty leve
       + +1000 · grabbed_now           # sparse, one-shot grande
```

**Grab detector v6**: solo `dist<0.03 AND lever>0.01 AND ball_z>0.10`. Eliminadas vel matching, z matching e indecision penalty.

**Otros recortes**:
- `r_floor_fail = 0` en todas las fases (sin penalty terminal por piso, solo terminación).
- `r_approach` complejo → reemplazado por `r_dist` lineal simple.
- `r_indecision`, `r_smooth`, `r_throw_*`, `r_hold`, `r_grab_height`, `r_grab_speed` → todos en 0. Constantes mantenidas por compat con el `info` dict.
- State machine (FALLING/HELD/THROWN) y connect equality se conservan en el código pero solo se usa FALLING + terminación al grab. Hold y throw quedan para iteraciones futuras.

**Por qué este orden**: si el agente no descubre el grab con este reward minimalista (que es el más permisivo posible para discovery), el problema no es de shaping sino de algoritmo (PPO con MlpPolicy + 256/256). En ese caso la siguiente jugada es SAC o ent_coef más alto. Si funciona, vamos agregando complejidad de a una capa: floor_fail, hold rewards, throw rewards.

**Primer experimento**: 5M steps en sub-fase 0 (bola estática). El objetivo es validar la hipótesis "el simple-reward sí permite discovery del grab".


## 2026-05-20 — v6, v6b: simple reward no destraba el grab

- **v6** (simple `r = -2*dist + 1000*grab`, ent_coef=0.005): la policy aprendió aproximación (14cm → 7cm) pero quedó atrapada ahí. Lever oscilando entre -1.4 y 1.8 mm, nunca cerrando. 0/12 grabs en eval determinístico. Confirmó que la info de la bola SI está en la obs (`ee_ball_delta` explícito), el problema no es de información sino de exploración.
- **v6b** (v6 + ent_coef=0.03 + bonus por `lever * (1 - dist/0.05)`): empeoró. La policy MÁS caótica no llegaba a <5cm para que el shaping del lever se active. ep_rew bajó a -110, ep_len permaneció en 251 (sin grab). El bonus de lever-close nunca disparó.

Diagnóstico: 7 variantes intentadas (v3, v4, v5b, v5c, v5d, v6, v6b). PPO con este reward landscape NO descubre el grab. La hipótesis del experto se confirmó: bottleneck es exploración, no shaping.

## 2026-05-20 — v7: pivot a "llevar de la mano" + curriculum diagonal

Después del fracaso del enfoque "explora la dinámica de caída", el usuario reorientó el setup completo. **Idea**: simplificar al máximo la tarea inicial para que la policy descubra el grab, después complicar gradualmente moviendo la bola.

**Cambios:**

1. **Bola amarilla sin colisión** (`contype="0" conaffinity="0"` en XML). La pelota es un "punto objetivo" virtual, no un cuerpo físico que choque con los dedos o el piso. Pasa a través de todo.
2. **Spawn nuevo**: justo arriba de la palma en reposo (`palm_rest + (0, 0, 3cm)`). Sin caída (gravedad ya estaba en 0 en subphase 0). La bola arranca en una posición casi siempre alcanzable por el grab.
3. **Curriculum diagonal**: después de `--curriculum-v7-after 1_000_000` steps, cada episodio el spawn se aleja 0.001m alternando **arriba** y **afuera** (-Y world). Cap en `CURRICULUM_OFFSET_MAX = 0.40 m`. Per-env counter (cada subproc actualiza el suyo).
4. **Detector de grab nuevo** (interpretación literal del spec "lever abriendo los dedos"):
   - `dist(EE, ball) < 0.04 m`
   - `lever_q < 0.003 rad` (dedos en posición abierta / default)
   - `ball_z > 0.10 m`
   - Si el usuario en realidad quería "lever cerrando", flippear `<` por `>` y subir umbral. Una línea en `_detect_grab`.
5. **Penalty terminal por timeout** sin grab: `r_floor_fail = -250` al `truncated=True`. Empuja a la policy a NO quedarse esperando.
6. **`r_lever_close` desactivado** (era shaping v6b que no aplica acá).

**Reward final v7:**
```
r = -2.0 * dist(EE, ball)            # dense, gradiente toward ball
  + -1e-4 * Σ τ²                     # effort
  + +1000 * grabbed_now               # one-shot success
  + -250 * truncated_without_grab     # one-shot fail at timeout
```

**Setup de training:**
- 3M steps total, 16 envs, `ent_coef=0.005`, subphase 0
- Curriculum diagonal activado a partir de step 1M (via callback `GrabCurriculumV7Callback`)
- Eval pos-training: 10 episodios con offsets variados (start/middle/end) + video MP4

**Bug encontrado durante calibración**:
- La equality `polycoef=34.9` del lever satura las articulaciones de los dedos en `lever_q ≥ 0.01`. Eso significa que `lever_q` efectivo está limitado a [0, 0.01]. El detector v5/v6 que pedía `lever > 0.01` JAMÁS disparaba (estaba justo en el límite saturado). Por eso parte del fracaso anterior.
- Adicionalmente, la dinámica acoplada del lever (forcerange ±1.08 N + damping reflejado de los dedos × 34.9²) hace que el lever tome ~11s para cerrar fully. Imposible en episodios de 5s. Pero como en v7 el grab pide `lever_q < 0.003` (dedos abiertos, posición default), este problema queda sidesteped.


## 2026-05-21 — v7b, v7c: curriculum per-episodio fracasa por catastrophic forgetting

- **v7b** (per-episode counter, +0.001m/ep alternating up/out, LAMBDA_DIST=2): con episodios de 1 step (grab trivial), la offset por env saturó a 0.40m en ~4000 episodios. Policy entrenó casi todo en max-offset → no generalizó. Eval mostró 3/10 grabs (todos en offset=0).
- **v7c** (LAMBDA_DIST=10, CURRICULUM_INCREMENT=0.0001, curriculum desde step 0): mismo bug pero peor. ep_rew_mean empezó +989 (grab trivial perfecto) y degradó a -641 con curriculum advance (catastrophic forgetting). Eval: offset=0 grab OK, offset=5cm policy se mueve solo 1cm (no llega), offset=15cm se mueve 12cm pero queda a 9-14cm de la bola.

**Causa raíz identificada**: el contador per-episodio sumaba al ritmo de los episodios, que cuando son cortos (1-step) crecen explosivamente. La distribución de offset durante el training se quedaba pegada al max_actual sin recordar el régimen easy.

## 2026-05-21 — Exploración manual del alcance del brazo

El usuario corrió `Scripts/explore_arm_reach.py` (visor MuJoCo interactivo con sliders + tracking de bounding box) para medir el alcance real de la palma en world frame, en vez de adivinarlo por la cinemática.

**Resultado del sweep manual:**
- `grip_R` baseline rest: (+0.0004, -0.0625, +0.1535)
- Max OUT (-Y desde rest): **0.1115 m** (11.15 cm)
- Max UP (+Z desde rest): **0.2112 m** (21.12 cm)
- Range X (lateral): solo 3.1 cm (la palma casi no se mueve lateralmente; era esperable porque shoulder solo rota frontal/dorsal)

Esto invalidó el cap de 0.40m simétrico que estaba usando v7b/v7c. El alcance real es **asimétrico** (mucho más Z que -Y) y mucho menor que lo asumido.

## 2026-05-21 — v7d: curriculum per-step + uniform sampling + caps empíricos

Consultado el experto, las 7 mejoras aplicadas:

1. **Cap asimétrico empírico**: `MAX_UP=0.19m` (90% de 21.1cm), `MAX_OUT=0.10m` (90% de 11.2cm). 10% de headroom para que el policy pueda extenderse mínimamente más allá del baseline observado.
2. **Curriculum per-STEP global** (no per-episodio): callback de SB3 lee `num_timesteps` y empuja `max_up` y `max_out` linealmente desde 0 hasta sus caps sobre el primer 70% del training. Esto evita que episodios cortos exploten la advance rate.
3. **Sampling uniforme `[0, max_t]` por reset**: cada episodio sortea `offset_up ∈ U[0, max_up]` y `offset_out ∈ U[0, max_out]` independientemente. La policy SIEMPRE ve un mix de difficulties → previene catastrophic forgetting.
4. **10% rehearsal a offset=0**: en cada reset, prob 0.10 de forzar offsets a cero (mantiene la skill trivial siempre presente en el rollout buffer).
5. **Shaped grab bonus** en los últimos 8cm: `+5·(1 - dist/0.08)` cuando dist<0.08. Da gradient denso en la zona crítica, sin distorsionar el shaping global.
6. **Timeout penalty bumped a -1000** (era -250). Con LAMBDA_DIST=10 y episodios de 250 steps, el -250 era dominado por el dist acumulado (-10·dist_avg·250 > 250 si dist_avg > 0.1). -1000 lo mantiene como señal relevante.
7. **PPO mantenido** (vs SAC que el experto desaconsejó para esta config: replay de SAC en CPU + curriculum no-estacionario = inestable).

**Reward final v7d:**
```
r_step = -10·dist
       + -1e-4·Σ τ²
       + 5·max(0, 1 - dist/0.08)   # solo dentro de 8cm
       + 1000 if grabbed_now
       + -1000 if truncated_no_grab
```

**Setup training**: 5M steps, 16 envs, ent_coef=0.005, subphase=0, `--curriculum-v7d` activado desde step 0 con caps asimétricos.


## 2026-05-21 — v7d: validación de la receta del experto

Aplicada la receta completa (caps asimétricos empíricos, uniform sampling, rehearsal, shaped bonus, timeout penalty bumped). Resultado: ep_rew_mean training ~997 estable, eval mostró **7/10 grabs**: 3/3 trivial, 4/4 medio (9cm up, 5cm out), 0/3 en el extremo max (19cm up, 10cm out — min_dist 4.96cm, fallaba por 1cm).

Diagnóstico: los caps al 90% del max empírico estaban demasiado cerca del límite cinemático real del brazo bajo control PPO.

## 2026-05-21 — v7e: cap al 80% + falling phase en los últimos 30%

Cambios sobre v7d:
- **Caps al 80%** del empírico (max_up=17cm, max_out=9cm) en vez de 90%.
- **Total 10M steps** (vs 5M en v7d).
- **Últimos 3M steps** (30% final): activar gravedad ligera g=-0.5 m/s² sobre la bola. La bola cae 25cm en 1s. Constante por episodio ("siempre igual" como pidió el usuario).
- **Progress log cada 2M steps** (el callback imprime curriculum status + ep_rew_mean / ep_len_mean).

Training corrió 1h26m (10M steps). El progreso ramp-up:
- step 2M (20%): max_up=4.9cm, max_out=2.6cm
- step 4M (40%): max_up=9.7cm, max_out=5.1cm
- step 6M (60%): max_up=14.6cm, max_out=7.7cm
- step 7M: FALLING ACTIVADO (g=-0.5)
- step 8M (80%): caps en máx, falling activo
- step 10M: ep_rew_mean ~875 (vs v7d 997, esperable por dificultad agregada)

**Eval final: 10/10 grabs**:
- STATIC trivial (off=0): 2/2 grab step 1
- STATIC medio (9 up, 5 out): 2/2 grab step 29
- STATIC max (17 up, 9 out): 2/2 grab step 91 — el caso que fallaba en v7d ahora pasa
- FALLING trivial (g=-0.5): 2/2 grab step 1
- FALLING max (17 up, 9 out, g=-0.5): 2/2 grab step 46-49 — la policy ANTICIPA la caída y la atrapa

ep_rew_mean=+991.16, ep_len_mean=33.9. Video en `runs/grab_phase1_v7e/eval_v7e_mixed.mp4`.

### Lecciones aprendidas (post-mortem de la fase grab)

1. **El reward simple gana**: 5 iteraciones complejas de reward shaping (v3-v5d) lograron 0 grabs. El reward minimalista `-2·dist + 1000·grab` (v6) consiguió aprender approach (aunque no cerrar el grab solo). El problema NO era la información en la obs ni el shaping fino — era la combinación de exploración insuficiente + curriculum mal diseñado.
2. **El curriculum per-step con uniform sampling es la diferencia**: las primeras versiones de curriculum (per-episodio) saturaban explosivamente cuando los episodios eran cortos. La key insight del experto fue: rampea el cap globalmente con `num_timesteps`, y dentro de cada reset samplea uniforme en [0, max_t]. Esto mantiene la distribución sobre el régimen easy siempre, previniendo catastrophic forgetting.
3. **Caps empíricos > geométricos**: no asumir el alcance del brazo. Sweep manual con `Scripts/explore_arm_reach.py` reveló que el palma alcanza 21cm en +Z pero solo 11cm en -Y — asimetría que invalidaba el plan original de cap simétrico a 40cm. Usar 80% del max empírico es defensible (margen para dinámica del policy).
4. **El shaped bonus en zona crítica vale la pena**: `+5·(1-dist/0.08)` en los últimos 8cm dio gradient denso justo donde el grab one-shot estaba "lejos" para PPO.
5. **PPO con CPU + 16 envs + MLP chica es la combinación correcta**: confirmado que GPU no aporta para este tamaño de red. Sample efficiency vino del curriculum y el shaping, no del algoritmo.

Próximo paso natural: agregar movimientos de hold + throw (fases 2 y 3 del state machine original que quedaron desactivadas en v6/v7). O escalar a tareas combinadas (grab + tracking).


## 2026-05-21 → 2026-05-22 — Integración head+arm+throw: v10, v11 (3 pasos del pedido del usuario)

El usuario pidió tres cosas: (1) que la cabeza siga la bola amarilla, (2) que la bola caiga de más alto y la policy del brazo vaya a buscarla, (3) que se pruebe throw. Se hicieron en cadena:

### Paso 1: Combined runtime head + arm (sin retraining)

`Scripts/run_combined_head_arm.py`: clase `CombinedHeadArmEnv(DUMGrabEnv)` que en cada `_apply_action` también:
- Computa la obs del head policy (replica `DUMHeadTrackingEnv._get_obs` operando sobre la MjData del grab env)
- Usa `data.qpos[ball]` como `target_world` (en vez del mocap rojo)
- Predice acción del head, inyecta su ctrl en los 3 actuadores `act_Neck`, `act_HeadBase`, `act_HeadRot`
- v13 no tenía `vecnormalize.pkl` — se pasa obs cruda

**Resultado**: v13 generaliza de "mocap rojo manual" a "bola amarilla seguida automáticamente". `avg_head_deg ≈ 20°` en evals (dentro de su entrenamiento σ=0.4 rad ≈ 23°). El head trackea sin retraining.

### Paso 2: v8 (fallido) → v10 (corregido)

- **v8**: 5M steps desde v7e, cap salto 17→30cm + g salto -0.5→-2.0. Catastrophic forgetting: el policy quedó "promediado", peor en todo lado. 1/6 grabs en eval, ep_rew_mean ~400 (vs v7e's 875).
- **v10**: 25M steps desde v7e, cap más conservador (17→25cm), gravity más suave (-0.5→-1.5), ramp 80% lineal. **15h41min de wall-clock por sleep de PC entre 02:00 y 14:46**. Eval: 3/6 grabs en distribución (rehearsal, small, mid). En high/MAX/STRESS el policy llega a min_dist=4-6cm pero no entra (4cm threshold).

### Paso 3: v11 throw enabled

- Re-habilitado state machine HELD→THROWN, pero NO se usa la `<connect>` equality (bug: con freejoint+contype=0 en MuJoCo 3.8 el solver da impulsos de 80cm al activar). En vez, override manual del `qpos` de la bola cada step en HELD para que siga la palma. Al transicionar a THROWN, la bola hereda la `qvel = ee_linvel` que se viene seteando cada step.
- Reward shaping para throw: `+500` grab one-shot (reducido vs +1000 de v7e), `+3` per-step en HELD (×125 steps = 375), `+600·tanh(dist_xy/0.5)` landing bonus, `-300` si cae detrás del torso.
- Auto-release a los 2.5s en HELD (no requiere que el policy lo decida).
- Training: 20M steps desde v10, 2h22m (sin sleep esta vez).

**Eval final** (v11 + v13, 8 episodios):
- rehearsal off=0: +358 (grab+hold OK, throw cae sin distancia porque g=0)
- small (8 up, 4 out, g=-1.5): **+1449** ciclo completo
- mid (15 up, 7 out, g=-1.5): **+1440** ciclo completo
- high/MAX/STRESS (≥20cm): -1040 cada uno (hereda problema de v10)

### Cierre honesto

**Lo que SI funciona end-to-end**:
- Head v13 trackea bola amarilla sin retraining
- Catch + Hold (2.5s) + Throw físicamente correcto (la bola sale con la velocidad del wrist al release)
- Las 3 trayectorias dentro de la distribución entrenada (rehearsal, small, mid) entregan ciclo completo con +1440 reward

**Lo que NO**:
- Offsets >20cm: el policy llega a 4-6cm de la bola pero no cierra (threshold 4cm). El v10 está al borde de hacer high pero no consistentemente.

**Razón estructural**: 5 horas de training efectivo (descontado el sleep). v7e había necesitado 15M+10M=25M de training acumulado para sus 10/10. v10 tiene 35M acumulados pero con un task significativamente más difícil (g=-1.5 vs g=-0.5, cap 25 vs 17cm).

Para que las configuraciones high/MAX entren consistentemente: harían falta otros 20-30M steps con la misma receta. La curva ya muestra signos positivos (min_dist 4-6cm es muy cerca, la policy SI intenta extender el brazo).


## 2026-05-22 → 2026-05-23 — v12, v13: throw natural + motor de animacion

### v12 (throw + curriculum gravity)
- 25M steps, resume v11, throw enabled
- Curriculum gravity ramp -1.5 → -6.0 sobre 80%, caps 25cm up / 10cm out
- Reward step durante THROWN: +10·forward_dist·dt
- Episode time bumped a 8s para tener tiempo de follow post-throw
- Resultado: 2/12 grabs (rehearsal + early g=-3.0 OK; mid+FINAL g=-6.0 fallaron — solo 5M steps a la dificultad MAX)
- ep_rew_mean ~530, ep_len ~90

### v13 (throw natural + cap extendido + color random)
- 25M steps, resume v12, throw enabled
- Cap extendido: up 25→40cm (60% mas que v12 estatico). out queda en 10cm.
- Gravity FIJA -2.5 (no ramp): la bola vuela mas tiempo, mas oportunidad de catch
- **NUEVO: bonus al release proporcional a velocidad forward del EE** (+200 pts por m/s). Encouragea el "swing" del brazo, no solo "drop"
- **NUEVO: color de la bola random per-episodio** (cosmetico, no afecta policy)
- Resultado: 4/12 grabs (mejor que v12). 2 de los 9 FINALes (cap MAX 40cm up) funcionan: eps 6 y 11 con ciclo completo +1392 a +1429.
- ep_rew_mean ~770, ep_len ~120 (ciclos completos)
- Visualmente el throw es MUCHO mas natural — el brazo acelera durante HELD antes del release, dando momento real a la bola

### Motor de animacion (runtime integrado, no requiere training)

`Scripts/run_animation_engine.py`: combina v13 head + v13 arm + state machine externo.

**State machine:**
- IDLE: head trackea bola roja (slider web), arm en pose neutra, bola amarilla a z=-10
- GRAB_CYCLE (al apretar boton `/grab_yellow` desde web):
  - Spawn bola amarilla en `palm_world + (0,0,0.40m)` con color random
  - Head switch target -> bola amarilla
  - Arm policy activa
  - Detecta grab interno -> HELD -> auto-release a 2.5s -> THROWN/FOLLOW (3s)
- Vuelta a IDLE

**Cancel parcial de gravedad post-release** (clave para "vuelo de 3s"):
- Durante THROWN/FOLLOW, `data.xfrc_applied[ball, z] = -0.80 * mass * gravity_z` (fuerza hacia arriba que compensa 80% del peso)
- Gravedad efectiva sobre la bola = 20% de la nominal (-2.5 → -0.5)
- La bola vuela mucho mas tiempo permitiendo que el head haga follow visible

**Web UI:**
- Boton "AGARRAR BOLA" agregado a `index.html`
- Telemetria muestra el estado del SM
- Endpoint POST `/grab_yellow` en `server.py`

Este es el "motor de animacion" del PLAN_ANIMATION_ENGINE: cada animacion (saludar, agarrar, etc) es un estado discreto del SM disparado por evento externo (boton). Las policies son skills primitivos.

### Cierre de la fase grab+throw integrada

Lo que entrega el proyecto:
- **Head tracking** funcional en cualquier target world (bola roja o bola amarilla, switch automatico)
- **Grab** consistente para offsets <=20cm en gravedad -2.5
- **Throw natural** con swing forward antes del release (no solo drop)
- **Animation engine** con state machine externo y trigger web — listo para extender con mas comportamientos

Lo que no esta perfecto: el grab al cap MAX (40cm up) funciona solo ~22% del tiempo en eval determinista. Mas training mejoraria. Pero el ciclo completo grab+hold+throw+follow YA es demostrable en los configs que SI funcionan.

## 2026-05-24 — v14 (rebalanced reward): policy de catch+throw production-ready

Resume desde v13 + reward shaping rebalanceado para premiar el throw lejos:
- LANDING_K: 600 → 1500 (saturacion a 3m en vez de 1m)
- VELOCITY_K: 200 → 500 (premia swing fuerte)
- THROW_STEP_K: 10 → 50 (per-step en THROWN)
- HOLD_PER_STEP: 3 → 1 (no quedarse quieto)
- LANDING_BACK_PENALTY: -300 → -500

Training: 50M steps, 6h. Resume v13. Sin cambios de cap (40/10) ni de gravity (-2.5). Eval: 3/12 grabs (rehearsal + 2 falling) — similar a v13 en catch rate, pero los throws exitosos entregaron +1430 vs +746. La policy SI aprendio a swingear forward antes del release. Mejor policy de catch hasta v16.

## 2026-05-24 → 2026-05-25 — v14_head, v14b, v14c: arreglando el "look atras"

Problema observado en v13_head: cuando el target estaba a azimuth > 90°, la cabeza ponia HeadBase perpendicular al piso en vez de usar HeadRotation_joint para yaw. Causa: v13 entreno con TARGET_CONE_AZIMUTH=75° — nunca vio targets atras.

**v14_head (resume v13 con curriculum) — FALLIDO**. Cono ramping 75°→150° azimuth en los primeros 7M de 20M steps + ent_coef bump 0.005→0.03 los primeros 5M. Para targets a 100°+, HeadRot saturado en ±45°. La policy se quedo en el local optimum heredado de v13.

**v14b (from scratch + light shaping) — Mejor pero no resuelto**. Training desde scratch, cono 150°/50° fijo, w_headrot_guidance=0.5: penalty si HeadRot no apunta al target_az. 30M steps. HeadRot mejoro algo pero HeadBase seguia compensando hasta ±40°.

**v14c (from scratch + strong shaping + HeadBase penalty) — FUNCIONO**. Setup de v14b pero con w_headrot_guidance=2.0 (4× mas fuerte) + w_headbase_tilt_penalty=1.5 (NUEVO: penalty DIRECTO a abs(HeadBase_qpos) cuando target_az > 30°) + net_arch 128,128. 30M steps, 3h33min. HeadRot trackea el target_az hasta ±160° (limite fisico), HeadBase queda en ±10°.

Bug clave del diagnostico: mi primer eval con 100-120 steps reportaba siempre "HeadRot saturado a ±45°". **Falso positivo**: el actuator HeadRot tiene forcerange=±2.11 + damping=5.0 → velocidad max ~0.42 rad/s. Para girar 160° necesita ~6.6s. Con 2-2.5s no llegaba. Subiendo a 400 steps (8s) el test mostro tracking real.

Scripts/eval_v14_head.py: script custom con targets en azimuth explicito y video con overlay (theta, HeadRot, HeadBase).

## 2026-05-25 — v15, v16: extension del workspace del arm (sin mejora)

**v15 (cap 50/15, curriculum static→falling) — FALLIDO**. Resume v14_long. Primera fase 0-40% con falling_active=False (bola estatica) + cap ramping 0→max(50/15). Segunda fase 40-100% falling g=-2.5 al MAX. 30M steps, 4h30min. Eval: 2/12 grabs (peor que v14).

Diagnostico: cap_up=50cm es FISICAMENTE INALCANZABLE para el brazo STATIC (sweep empirico mostro max +21cm en Z). La fase static (12M steps) entreno con targets imposibles → catastrophic forgetting de v14.

**v16 (cap 35/15, sin fase static) — sin mejora**. Resume v14_long (NO desde v15). CURRICULUM_MAX_UP=0.35, MAX_OUT=0.15. SIN fase static — falling g=-2.5 desde step 0. 30M steps, 4h13min. Eval: 2/12 grabs (igual que v15). Plateau confirmado: con esta cinematica + PPO + reward shape, ~3/12 grabs es el techo.

Decision: rollback al v14_long como mejor arm policy para el animation engine. Las mejoras del catch necesitan otras herramientas (SAC, modelo con actuadores mas fuertes, reward shaping nuevo). Quedan para futuras iteraciones.

## 2026-05-25 → 2026-05-26 — Animation engine: integracion final

Scripts/run_animation_engine.py con defaults production-ready:
- arm: runs/grab_phase1_v16_extended/final.zip (default, fallback a v14_long)
- head: runs/ppo_dum_v14c_aggressive/final.zip
- Autofoco procedural (replica de _lens_focus_ctrl del head env)
- Web server FastAPI en thread daemon
- UI landscape: stream MJPEG izquierda 50%, acciones derecha scrollable

**Saludo procedural integrado** (Scripts/rl/procedural/wave.py). Pre-existia el modulo WaveAnimation (4 fases: retract → wrist half-turn → fingers cycle×2 → return) pero no estaba wired. Agregado: estado STATE_WAVING en el SM externo, endpoint POST /wave, boton SALUDAR en index.html, override de los 4 actuadores del brazo elegido cuando wave activo. Lado random R/L por click. WAVING mutuamente exclusivo con grab cycle.

**v2 del wave (despues de prueba inicial)**. Wave inicial casi invisible: amplitudes chicas + timing rapido para actuadores debiles (wrist forcerange=±0.21 N·m). Bumped: DURATION_S 4→6s, FOREARM_RETRACT 0.65→0.95, WRIST_HALF_TURN 0.75→1.00, ventanas estiradas proporcionalmente.

### Bug fixes al animation engine

1. **Cabeza se "caia" durante WAVING**: head_target estaba seteado a ball_pos cuando no era IDLE. La bola amarilla en WAVING esta a z=-10 (escondida) → head trataba de mirar al underground. Fix: head_target = red_target_world para todos los estados que no sean grab cycle.

2. **target_vel explosivo al spawn de la bola**: cuando se triggerea grab_yellow, la bola salta de z=-10 a z=palm+0.4. Siguiente step computa target_vel ~500 m/s. Out of distribution. Fix: tras spawn_yellow_ball, reset target_world_prev = new_ball_pos.

3. **Boton se rehabilitaba antes del fin del ciclo**: timeout fijo 2500ms vs ciclo grab+throw de 6-8s. Fix: el JS lee state de la telemetria WebSocket y habilita/deshabilita botones segun el state real (IDLE = activos, cualquier otro = deshabilitados con label "esperá: STATE").

### Limites del target del head (anti-out-of-distribution)
v14c entreno con target distance 0.3-1.0m del lens. El slider podia mandar targets a 1.27m con combinaciones extremas. Agregado: DIST_MIN -0.65, DIST_MAX +0.95, y un Clamp 3D (clampTargetToTrainingDist en control.js) que proyecta el target sobre el rayo lens→target para que la distancia quede en [0.30m, 1.00m]. Aunque el usuario combine valores extremos, el head nunca recibe un target out-of-distribution.

## Estado actual del sistema (cierre de sesion)

### Policies entrenadas
- **v13_head** (3M): tracking ±75° az. Reemplazado por v14c.
- **v7e** (15M acumulado): catch baseline 10/10 en config fija. Base para extensiones.
- **v14_long** (50M acumulado): catch+throw, reward rebalanceado. Mejor catch absoluto.
- **v14c_aggressive** (30M from scratch): head tracking ±150° az con HeadBase estable. Mejor head.
- v15, v16: intentos de extension del workspace de catch, no superaron a v14_long.

### Pipeline production-ready
```
runs/grab_phase1_v16_extended/final.zip   ← arm (default animation engine)
runs/ppo_dum_v14c_aggressive/final.zip    ← head (default animation engine)
Scripts/run_animation_engine.py            ← runtime integrado web + viewer
Scripts/web_remote/                        ← server FastAPI + UI landscape
Scripts/rl/procedural/wave.py              ← saludo procedural (sin RL)
```

### Para usar
```
py -3 Scripts/run_animation_engine.py
```
Browser → http://localhost:8000 (landscape):
- Canvas + slider: mover bola roja (target del head, clamp 3D al rango de training)
- Boton AGARRAR BOLA: grab+hold+throw+follow (~6s, brazo random)
- Boton SALUDAR: saludo procedural 6s (brazo random)
- Estado del SM visible abajo del stream
- Botones habilitan SOLO en IDLE (segun telemetria real)

### Pendientes (PLAN_NEXT_TRAININGS.md)
- Mejorar catch al cap MAX 40cm (requiere SAC o modelo con actuadores mas fuertes)
- Sim-to-real (proyecto final entrega solo simulacion validada)

---

# Reward functions finales

## Head policy (v14c_aggressive)

Definida en `Scripts/rl_env.py` (clase `DUMHeadTrackingEnv`). Por step:

```
r = w_track       · exp(-theta² / sigma²)                                 [tracking core]
  − w_smooth      · |a_t − a_{t-1}|²        · focus_mult                  [accion suave]
  − w_effort      · |a_t|²                                                 [poco torque]
  − w_jitter      · |qvel_head|²            · focus_mult                  [no temblar]
  + w_alive                                                                [bonus por step vivo]
  + w_focus_event * (focus_just_triggered ? 1 : 0)                        [eliminado en v14c]
  + w_focus_hold  · sqrt(focus_hold_streak) (si theta < FOCUS_THRESHOLD)  [premia hold prolongado]
  − w_headrot_guidance      · |HeadRot_qpos − target_az|                  [si |target_az|>17°]
  − w_headbase_tilt_penalty · max(0, |HeadBase_qpos| − 0.2 rad)           [si |target_az|>30°]
```

**Pesos finales (v14c_aggressive):**

| Peso | Valor | Que mide / por que |
|---|---|---|
| `w_track` | 1.0 | Tracking core. `theta` = angulo entre lens_forward y target_dir. Exp con sigma=0.4 → r=1 cuando alineado, r=0.97 a 7°, r=0.29 a 28°, r=0.08 a 45°. |
| `sigma` | 0.4 rad | Ancho del exponencial de tracking. Mas grande = gradient mas suave lejos del target. |
| `w_smooth` | 0.15 | Penaliza cambios bruscos en la accion |a_t − a_{t-1}|². Multiplicado por focus_mult cuando ya esta en target (premia "quietud" cuando enfoca). |
| `w_effort` | 0.0075 | Penaliza accion grande |a_t|². Mantiene low-energy. |
| `w_jitter` | 0.01 | Penaliza qvel residual de la cabeza |qvel_head|². Es lo que da ESTABILIDAD visual — sin esto la cabeza vibra. Tambien con focus_mult. |
| `w_alive` | 0.0 | Bonus por step (deshabilitado). |
| `w_focus_event` | 0.0 | One-shot al disparo del focus (eliminado por el experto: escalon discreto rompe value function). |
| `w_focus_hold` | 0.03 | Premio creciente con sqrt(streak_steps) mientras theta<umbral. Premia mantener el target prolongado sin "explotar" la magnitud. |
| `w_post_focus_mult` | 1.5 | Multiplicador a smooth+jitter cuando el focus ya disparo. Refuerza la "quietud" post-enfoque. |
| **`w_headrot_guidance`** | **2.0** | **NUEVO v14c**. Penalty proporcional al mismatch entre HeadRot_qpos y target_az esperado. Solo se aplica si abs(target_az) > 0.3 rad (17°). Empuja a usar HeadRotation_joint (yaw) en vez de compensar con HeadBase. |
| **`w_headbase_tilt_penalty`** | **1.5** | **NUEVO v14c**. Penalty directo a abs(HeadBase_qpos) cuando target_az es grande. Aplica si abs(target_az) > 0.5 rad (30°), penaliza solo el EXCESO sobre 0.2 rad. Fuerza a NO usar HeadBase para compensar yaw — HeadBase queda libre para elevation moderada del target. |

`FOCUS_THRESHOLD = 0.07 rad ≈ 4°` (umbral para considerar "enfocado").

## Arm policy (v16_extended, hereda de v14_long)

Definida en `Scripts/rl/envs/grab_env.py` (clase `DUMGrabEnv`). State machine FALLING → HELD → THROWN. Por step:

```
r = r_dist + r_effort + r_grab_first + r_lever_close + r_floor_fail
  + r_hold + r_throw_landing + r_throw_dist + r_throw_velocity
```

Cada termino depende del estado de la maquina:

| Termino | Aplica cuando | Formula | Valor / peso |
|---|---|---|---|
| `r_dist` | siempre | `−LAMBDA_DIST · dist(EE, ball)` | `LAMBDA_DIST = 10.0` — gradient denso para que la palma persiga la bola. dist en metros. |
| `r_effort` | siempre | `−LAMBDA_EFFORT · Σ tau_act²` | `LAMBDA_EFFORT = 1e-4` — penalty leve por torque cuadratico de los actuadores del brazo. Mantiene low-energy sin frenar el movimiento. |
| `r_grab_first` | one-shot al detectar grab | `+THROW_GRAB_BONUS` | `THROW_GRAB_BONUS = 500.0`. Disparado cuando dist(EE, ball)<0.03m AND lever_q<0.003 rad AND ball_z>0.10. |
| `r_lever_close` | en FALLING, dist<0.08m | `SHAPED_BONUS_MAG · (1 − dist/SHAPED_BONUS_DIST)` | `SHAPED_BONUS_MAG = 5.0`, `SHAPED_BONUS_DIST = 0.08m`. Gradient denso en los ultimos 8cm para evitar que el grab one-shot quede como reward sparse. |
| `r_floor_fail` | terminal, bola toca piso sin agarrar O timeout en Fase 1 | `−P_TIMEOUT_NO_GRAB` | `P_TIMEOUT_NO_GRAB = 1000.0`. Castigo terminal si el episodio termina sin grab. |
| `r_hold` | en HELD, per-step | `+THROW_HOLD_PER_STEP` | `THROW_HOLD_PER_STEP = 1.0` (reduced de 3.0 en v14 para no incentivar "quedarse quieto sosteniendo"). Sobre HELD ~125 steps → ~125 pts. |
| `r_throw_landing` | one-shot al touchdown en THROWN | `+THROW_LANDING_K · tanh(dist_xy_base / THROW_LANDING_DIST_SCALE)` o `THROW_LANDING_BACK_PENALTY` si cayo atras | `THROW_LANDING_K = 1500.0`, `THROW_LANDING_DIST_SCALE = 1.5m` (saturacion a 3m), `THROW_LANDING_BACK_PENALTY = −500.0`. Premia tirar lejos hacia adelante; castiga tirar atras. |
| `r_throw_dist` | en THROWN, per-step | `+THROW_STEP_K_FORWARD · max(0, ee_y − ball_y) · dt` | `THROW_STEP_K_FORWARD = 50.0` per metro/segundo. Premio continuo durante el vuelo: bola alejandose hacia adelante del robot. forward = −Y world. |
| `r_throw_velocity` | one-shot al release | `+THROW_RELEASE_VELOCITY_K · max(0, −ee_v_y)` | `THROW_RELEASE_VELOCITY_K = 500.0` por m/s. Premia que la EE este SWINGEANDO forward al momento del release (no solo "soltar quieto"). Encouragea movimiento de throw natural. |

**Otros parametros relevantes del shaping:**

- `GRAB_DIST_THR = 0.03 m` — umbral de contacto EE/bola para detectar grab.
- `GRAB_LEVER_OPEN_THR = 0.003 rad` — lever debe estar "abierto" (~0) para que el grab cuente.
- `THROW_HELD_MAX_S = 2.5 s` — tras 2.5s en HELD, auto-release con la velocidad del EE en ese instante.
- `EPISODE_MAX_TIME_THROW = 8.0 s` — episode timeout cuando throw enabled.
- Curriculum del cap (v16): `CURRICULUM_MAX_UP = 0.35 m`, `CURRICULUM_MAX_OUT = 0.15 m` — ramp lineal 0 → max sobre los primeros 30% del training, luego mantiene.
- Gravedad: `FALLING_GRAVITY = -2.5 m/s²` constante (no es realista, pero permite tiempo de reaccion suficiente para PPO).

---

# Estimacion de motores reales y costos (a partir de los torques simulados)

Mapeo de los `forcerange` finales del MJCF a servos comerciales. Conversion: kg·cm = N·m / 0.0980665. Para `act_HipBody` el torque de junta = gear(3) × forcerange(12.11) = 36.3 N·m.

## Actuadores y torque simulado

| Actuador | forcerange (N·m) | gear | Torque junta | kg·cm |
|---|---|---|---|---|
| act_HipBody | ±12.11 | 3 | ±36.3 N·m | ±370 |
| act_Neck / HeadBase / HeadRot | ±2.11 | 1 | ±2.11 N·m | ±21.5 |
| act_Left/RightShoulderArm | ±2.11 | 1 | ±2.11 N·m | ±21.5 |
| act_Left/RightForearm | ±2.11 | 1 | ±2.11 N·m | ±21.5 |
| act_BodyShoulderLeft/Right | ±1.08 | 1 | ±1.08 N·m | ±11.0 |
| act_Left/RightLever_Slider | ±1.08 | 1 | ±1.08 (lineal, leva) | ~11.0 |
| act_Left/RightWrist | ±0.21 | 1 | ±0.21 N·m | ±2.14 |
| act_LenteExt | ±0.21 | 1 | ±0.21 (lineal, foco) | ~2.14 |

15 actuadores en total.

## Agrupacion por familia de servo + cantidades

| Familia (torque) | Servo de referencia | Actuadores | Cant. |
|---|---|---|---|
| 21.5 kg·cm | DS3218 (20-21.5 kg·cm metal) | Neck, HeadBase, HeadRot, 2× ShoulderArm, 2× Forearm | 7 |
| 11 kg·cm | MG996R (9.4-11 kg·cm) | 2× BodyShoulder, 2× Lever (leva) | 4 |
| 2.2 kg·cm | MG90S (1.8-2.2 kg·cm micro) | 2× Wrist, LenteExt (foco) | 3 |
| 370 kg·cm (caso especial) | ver abajo | HipBody | 1 |

## Caso especial: act_HipBody (36 N·m)

Outlier (rota todo el torso + brazos + cabeza), ya marcado como problematico en el proyecto. 370 kg·cm esta muy por encima de cualquier servo hobby (el mas grande, DS5160, da ~65 kg·cm). Opciones:
- NEMA 23 + caja planetaria 20:1 (stepper 1.9 N·m × 20 = 38 N·m): USD 70-120, pesado, necesita driver.
- DS5160 (6.4 N·m) + reduccion 6:1: USD 45 servo + 20-40 reduccion, hay que fabricar la reduccion.
- Dynamixel MX-106/XM540 (8-10 N·m) + 4:1: USD 300-500, caro.

NOTA: conviene revisar si los 36 N·m son reales o estan sobredimensionados en sim (masa PLA + kp=800 rigido pueden inflar el requerimiento). Si el torso real es mas liviano, alcanzaria DS5160 + reduccion 6:1 (~USD 65-85).

## Bill of materials (motores) — precios USD referencia (AliExpress/calle; ARG +50-100% por importacion)

| Familia | Modelo | Cant. | Unit. | Subtotal |
|---|---|---|---|---|
| Torso/cuello/brazos | DS3218 | 7 | ~16 | ~112 |
| Hombros-cuerpo + levas pinza | MG996R | 4 | ~5 | ~20 |
| Muñecas + foco | MG90S | 3 | ~4 | ~12 |
| Cadera | NEMA23+planetaria o DS5160+reduccion | 1 | 70-120 | ~70-120 |
| **TOTAL** | | **15** | | **≈ USD 215-265** |

## Caveats
1. Torques de SIMULACION (sin friccion real, backlash, picos dinamicos). Agregar 30-50% margen → grupo 21.5 kg·cm subir a DS3225 (25 kg·cm).
2. Wrists (0.21 N·m): MG90S da justo 2.2 kg·cm, esta AL LIMITE. Si la muñeca sostiene la bola en vuelo, subir a micro de 3-4 kg·cm.
3. Levas y foco son juntas slide (lineales): el torque real del servo depende del radio de la leva impresa, hay que verificarlo con la geometria.
4. NO incluye: fuente/LiPo dimensionada para corriente de stall, reguladores/BEC, 1-2 placas driver de servos (PCA9685 ~USD 3 c/u).
