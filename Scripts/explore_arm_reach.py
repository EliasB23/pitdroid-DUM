"""Visor interactivo del DUM4_grab.xml para explorar el alcance de los brazos.

Abre el MuJoCo passive viewer con el modelo cargado. Mientras esta abierto:
- En la barra LATERAL del viewer (Control panel), podes arrastrar los sliders de
  cada actuador para mover los joints en tiempo real.
- En la TERMINAL se imprime cada 0.5s la posicion world de las palmas (sites
  grip_R y grip_L) y su offset respecto al baseline (pose en reposo).
- Al cerrar el viewer, se imprime el BOUNDING BOX maximo que las palmas
  alcanzaron durante la sesion (util para definir el cap de la curriculum).

Tips del viewer (panel izquierdo "Control"):
- Slider 'act_HipBody'        : cadera (rotacion del torso)
- Slider 'act_*ShoulderArm'   : hombros
- Slider 'act_*Forearm'       : codos
- Slider 'act_*Wrist'         : munecas
- Slider 'act_*Lever_Slider'  : pinza

Si no ves el panel Control, presiona la tecla "Tab" en el viewer.

Uso:
    python Scripts/explore_arm_reach.py
"""
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer


XML_PATH = str(Path(__file__).resolve().parent.parent / "Cuerpo" / "DUM4_grab.xml")
print(f"[setup] Cargando: {XML_PATH}")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
print(f"[setup] DOFs: {model.nq} qpos, {model.nu} actuadores, dt={model.opt.timestep}s")

# IDs de los sites de las palmas
grip_R = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_R")
grip_L = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_L")
if grip_R < 0 or grip_L < 0:
    raise RuntimeError("No encontre los sites grip_R / grip_L en el XML")

# Settling — dejar que el robot se asiente en su pose default
mujoco.mj_forward(model, data)
for _ in range(50):
    mujoco.mj_step(model, data)

baseline_R = data.site_xpos[grip_R].copy()
baseline_L = data.site_xpos[grip_L].copy()
print()
print("[baseline] Posiciones de palma EN REPOSO (world frame):")
print(f"   grip_R: ({baseline_R[0]:+.4f}, {baseline_R[1]:+.4f}, {baseline_R[2]:+.4f})")
print(f"   grip_L: ({baseline_L[0]:+.4f}, {baseline_L[1]:+.4f}, {baseline_L[2]:+.4f})")
print()

# Bounding box tracker
bbox_R_min = baseline_R.copy()
bbox_R_max = baseline_R.copy()
bbox_L_min = baseline_L.copy()
bbox_L_max = baseline_L.copy()

print("[viewer] Abriendo viewer. Cerralo cuando termines de explorar.")
print("[viewer] Mové los sliders del panel 'Control' para articular el brazo.")
print()
print(f"{'time':>6s}  {'grip_R x':>9s} {'y':>9s} {'z':>9s}  |  "
      f"{'offset_x':>9s} {'offset_y':>9s} {'offset_z':>9s}  |  {'dist_xyz':>9s}")

last_print = time.time()
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        # Update bbox
        pR = data.site_xpos[grip_R]
        pL = data.site_xpos[grip_L]
        bbox_R_min = np.minimum(bbox_R_min, pR)
        bbox_R_max = np.maximum(bbox_R_max, pR)
        bbox_L_min = np.minimum(bbox_L_min, pL)
        bbox_L_max = np.maximum(bbox_L_max, pL)
        # Print every 0.5s
        now = time.time()
        if now - last_print >= 0.5:
            off = pR - baseline_R
            dist = float(np.linalg.norm(off))
            print(f"{data.time:6.2f}  {pR[0]:+9.4f} {pR[1]:+9.4f} {pR[2]:+9.4f}  |  "
                  f"{off[0]:+9.4f} {off[1]:+9.4f} {off[2]:+9.4f}  |  {dist:+9.4f}",
                  flush=True)
            last_print = now
        time.sleep(model.opt.timestep)

# Final summary
print()
print("=" * 70)
print("BOUNDING BOX (durante la exploracion)")
print("=" * 70)
for side, bmin, bmax, base in [("grip_R", bbox_R_min, bbox_R_max, baseline_R),
                                ("grip_L", bbox_L_min, bbox_L_max, baseline_L)]:
    print(f"\n{side} (baseline rest: {base[0]:+.4f}, {base[1]:+.4f}, {base[2]:+.4f})")
    print(f"  X world: [{bmin[0]:+.4f}, {bmax[0]:+.4f}]  rango {bmax[0]-bmin[0]:.4f}m  "
          f"offset: [{bmin[0]-base[0]:+.4f}, {bmax[0]-base[0]:+.4f}]")
    print(f"  Y world: [{bmin[1]:+.4f}, {bmax[1]:+.4f}]  rango {bmax[1]-bmin[1]:.4f}m  "
          f"offset: [{bmin[1]-base[1]:+.4f}, {bmax[1]-base[1]:+.4f}]")
    print(f"  Z world: [{bmin[2]:+.4f}, {bmax[2]:+.4f}]  rango {bmax[2]-bmin[2]:.4f}m  "
          f"offset: [{bmin[2]-base[2]:+.4f}, {bmax[2]-base[2]:+.4f}]")

print()
print("Para el curriculum (relevante para R, mirrored para L):")
max_out_R = max(0.0, baseline_R[1] - bbox_R_min[1])  # outward = -Y
max_up_R = bbox_R_max[2] - baseline_R[2]
print(f"  max OUTWARD (-Y): {max_out_R:.4f}m")
print(f"  max UP (+Z):      {max_up_R:.4f}m")
print(f"  cap simetrico sugerido: {min(max_out_R, max_up_R):.4f}m")
