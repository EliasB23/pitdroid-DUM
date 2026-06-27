# Plan de Diseño Técnico — Animation Engine DUM

> **Generado por agente experto. Para implementar después del MJPEG.**
> Arquitectura inspirada en BD-X (Disney RSS 2024) y Olaf (Disney 2025).

## Arquitectura general (estilo BD-X)

```
                    AnimationEngine (50 Hz, mux a ctrl[15])
                    ┌─────────────────────────────────────┐
WS events ─event──▶ │ StateMachine: IDLE / WAVE / GRAB    │
                    └──────┬──────────────────────────────┘
                           │ por step, llena ctrl[15] componiendo:
   ┌───────────────────────┼────────────────────────────────┐
   │ Layer A: TRACKING (v13, RL perpetual)                  │
   │   owns: [Neck, HeadBase, HeadRot]                      │
   │   target: red_mocap (default) | yellow_ball (en GRAB)  │
   ├────────────────────────────────────────────────────────┤
   │ Layer B: SALUDAR (procedural, sinusoidal + crossfade)  │
   │   owns: {L|R}{ShoulderArm, Forearm, Wrist}             │
   ├────────────────────────────────────────────────────────┤
   │ Layer C: GRAB (RL episódico, on-demand)                │
   │   owns: {L|R}{ShoulderArm, Forearm, Wrist, Lever}      │
   ├────────────────────────────────────────────────────────┤
   │ Layer D: DEFAULT (resto: BodyShoulders, HipBody, Lente)│
   └────────────────────────────────────────────────────────┘
```

**Mux por ownership de actuadores, no por prioridad global**: capas escriben en índices disjuntos → no compiten. Tracking siempre activo.

## Saludar (procedural, sin training)

Trayectoria sinusoidal sobre brazo (3 ciclos, T=1.2s, f=0.83Hz):
```
θ_shoulder(t) = θ_sh_base + 0.9                       # eleva ~52°
θ_forearm(t)  = θ_fa_base + 0.3·sin(2π·t/T)
θ_wrist(t)    = θ_wr_base + 0.6·sin(2π·t/T + π/4)
```
Crossfade cuadrático 0.4s in/out. Duración total ≈ 4s.

## DUMGrabEnv (RL episódico)

**Action (4 dims):** [shoulder, forearm, wrist, lever_slider] del brazo elegido.

**Obs (21 dims):**
- Ball pos en frame head (3) + ball vel local (3) + ball z absoluto (1)
- EE pos en frame head (3) + EE−ball delta (3)
- qpos arm (4) + qvel arm (3) + lever vel (1)

**Reward:**
```
r_approach = exp(-‖EE − ball‖² / 0.05)              # σ≈22cm
r_grab     = +50  si lever > 0.008 ∧ ‖EE−ball‖ < 0.06
r_floor    = −30  si ball_z < 0.05
r_smooth   = −0.01 · ‖Δa‖²
r_effort   = −0.001 · ‖a‖²
r_alive    = −0.05                                   # resolver rápido
```

**Bola amarilla**: agregar al XML con `<freejoint>` + gravedad real (mejor que mocap fake).

**Cabeza durante grab**: conmutar `target_source` de la roja a la amarilla. v13 sigue corriendo, cero re-training. (Opción C del experto, simplísima.)

**Reset**: lado random (o specialist por lado). Bola en x = ±0.35m, y = 0.25m, z ∈ U(1.6, 2.0)m. Cae por gravedad.

**Terminación**: éxito (grab) o fail (z<0.05) o truncado a 8s.

## Curriculum (CRÍTICO para que converja)

| Fase | Steps | Setup |
|---|---|---|
| 1 | 1M | Bola **estática** a altura agarrable |
| 2 | 1M | Caída lenta (g=2 m/s²) |
| 3 | 1M | Caída real (g=9.81) |

Sin curriculum, PPO no converge por sparsity del éxito.

## Hyperparams training

| Param | Valor |
|---|---|
| Algo | PPO (SB3) |
| n_envs | 16 (SubprocVecEnv) |
| total_timesteps | 3M (un brazo) o 5M (mirror aug) |
| n_steps | 2048 |
| batch_size | 512 |
| net_arch | [256, 256] |
| lr | 3e-4 → linear decay |
| ent_coef | 0.005 |

**Wall-clock**: 4-6h CPU 8 cores.

## Cambios al XML

1. Yellow ball con freejoint + gravedad.
2. Sites `grip_L` y `grip_R` en los wrist/lever para detectar EE.
3. NO equality conditional — manejar "agarre" con teleport+attach en Python.

## Integración runtime

```
WS event "grab" →
  AnimationEngine.state = GRAB
  target_source = yellow_ball              # v13 sigue la amarilla
  spawn ball (freejoint reset)
  Layer C (grab policy) ↔ ctrl[arm+lever]
  on done: target_source = red; state = IDLE
```

Transición suave: blend lineal 0.3s entre ctrl_prev y policy_grab al entrar.

## Comparación con BD-X y honestidad

**Similar**: layered engine, perpetual+episodic, crossfade de animations.
**Más ambicioso**: BD-X "jump"/"happy dance" son gestos cerrados; **agarrar bola que cae requiere predicción + timing**. Más cerca de papers como:
- *Catching objects in flight* (Kim 2014)
- *DeepMind OP3 catching* (2021)
- *Agile Catching with Whole-Body MPC* (ETH 2023)

**Es razonable** con curriculum. Demo en simulación impactante; transferencia real al hardware queda como future work.

## Orden recomendado (1.5 semanas)

1. **Día 1**: SALUDAR procedural (victoria rápida, sin RL).
2. **Día 2-3**: DUMGrabEnv con bola ESTÁTICA (validar pipeline).
3. **Semana 2**: Curriculum completo, 3 fases.

## Estructura de archivos sugerida

```
Scripts/
├── rl/
│   ├── envs/
│   │   ├── grab_env.py
│   │   └── grab_eval_helpers.py
│   ├── procedural/
│   │   └── wave.py
│   ├── animation_engine.py
│   └── train_grab.py
└── ... (resto sin tocar)
```
