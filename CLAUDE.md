# DUM_MJC — Robot Animatrónico (Proyecto Final)

Proyecto Final de Carrera del Técnico Universitario en Mecatrónica. Robot animatrónico inspirado en el **Pit Droid** de Star Wars: diseñado en CAD, simulado en MuJoCo, y con perspectiva futura de impresión 3D en PLA.

Esta carpeta (`DUM_MJC/`) es el "paquete de simulación": contiene el modelo MJCF, las mallas STL y los scripts de Python para validar la dinámica y calibrar los actuadores.

---

## Objetivos del proyecto

1. **Aprender el flujo profesional** completo de creación de un animatrónico: idea → CAD → exportación → simulación física → control.
2. **Modelar en CAD** un personaje reconocible con técnicas de parametrización y simetría (Fusion360).
3. **Exportar a MJCF** mediante el plugin ACDC4Robot, generando un `xml` válido para MuJoCo más sus mallas STL.
4. **Corregir el modelo exportado**: unidades (grados → radianes), límites de joints, propiedades de actuadores, restricciones de igualdad (leva → dedos).
5. **Simular y calibrar** los 13 actuadores de posición con búsqueda en grilla sobre `kp`, `kv`/`kd`, `damping` y `gear`, evaluando estabilidad, sobreimpulso y precisión.
6. **Documentar todo el proceso** en el `.docx` de la carpeta padre, para la presentación final.

La construcción física queda como horizonte; el alcance entregable es el modelo simulado y controlado.

---

## Estructura

```
DUM_MJC/
├─ Cuerpo/
│  ├─ DUM4.xml              # Modelo MJCF principal
│  └─ meshes/*.stl          # 21 mallas de los "links" del robot
└─ Scripts/
   ├─ calibracion.py        # Grid search paralelo (multiprocessing)
   ├─ forzar_motores.ipynb  # Notebook tutorial / banco de pruebas
   └─ requirements.txt      # mujoco, numpy, matplotlib, jupyter, pandas
```

Archivos hermanos relevantes en `../` (carpeta padre `ProyectoFinal/`):
- `Proyecto CREACIÓN DE ROBOT ANIMATRÓNICO YYYYMMDD.docx` — documento del proyecto (varias versiones fechadas; la más reciente es la canónica).
- `Diario de desarrollo.docx` — diario.
- `Seguimiento_Kanban.xlsx` — tablero Kanban.
- `Modelo3D/`, `Cuerpo/` (otros), `DUM4/` — historiales/exportaciones anteriores del CAD.

---

## Stack técnico

| Capa | Herramienta |
|---|---|
| CAD | **Autodesk Fusion360** (licencia educativa) |
| Exportador CAD → MJCF | Plugin **ACDC4Robot** para Fusion360 |
| Simulador | **MuJoCo** (binario Windows y API Python) |
| Lenguaje | **Python 3** |
| Entorno | **Jupyter Notebook** + scripts `.py` |
| Librerías | `mujoco`, `numpy`, `matplotlib`, `pandas` |
| Paralelización | `multiprocessing.Pool` para grid search por actuador |
| Plataforma | Windows 10 |
| Materiales (planeados) | **PLA** (impresión 3D) — densidades reales asignadas en Fusion360 para que la simulación use masas e inercias realistas |
| Electrónica (planeada) | Raspberry Pi 5 + baterías LiPo |
| Servomotores (referencia para `forcerange`) | **DS3218** (torso/cuello, 21.5 kg·cm), **MG996R** (brazos/hombros, 11 kg·cm), **MG90S** (muñecas/leva, 2.2 kg·cm) |

---

## Reglas y convenciones del modelo MJCF

### Unidades
- **Ángulos en radianes** (`<compiler angle="radian"/>`). Conversión: `1° = π/180 ≈ 0.01745329252 rad`. Fusion360 trabaja en grados — al pasar al `xml` hay que convertir cualquier límite manualmente.
- **Mallas en metros**: las STL están en milímetros, se escalan con `scale="0.001 0.001 0.001"` al cargarlas.
- **Torques en N·m**: los datasheets de servomotores suelen estar en kg·cm. Conversión: `N·m = kg·cm × 0.0980665`.

### Nomenclatura
- Patrón: `[Lado][Parte]_[tipo]`. Ejemplos: `RightTopFinger_joint`, `LeftShoulderArm_Link`, `act_RightWrist`.
- Sufijos: `_link` para cuerpos, `_joint` para articulaciones, `_geom` para geometrías visuales, `act_<Joint>` para actuadores.

### Agrupación en "links"
- Un *link* agrupa todas las piezas físicas que se mueven juntas entre dos `joint`s. Esto reduce los cuerpos que el solver debe trackear y simplifica el cálculo de masas/inercias.
- Cada link tiene material **PLA** en Fusion360 → masas e inercias del MJCF reflejan el peso real estimado del robot impreso.

### Joints
- Siempre `limited="true"` + `range="min max"` (en radianes) para respetar los topes mecánicos del diseño CAD.
- Tipos usados: `hinge` (la mayoría) y `slide` (levas de los dedos y mecanismo de lente).
- `frictionloss` y `damping` se completan donde aplica.

### Restricciones de igualdad (leva → dedos)
- Mecanismo: un servo con leva (hinge → slider) empuja los dedos (hinge); en el simulador la relación mecánica se reemplaza por una `<equality joint .../>` con polinomio lineal.
- Coeficiente: `polycoef="0 ±34.906585 0 0 0"`. Surge de `max_rango_dedos / max_rango_leva = 0.34906585 rad / 0.01 m`.
- Signo positivo para el dedo superior, negativo para el inferior (giran en sentidos opuestos).
- Aplica a las 4 articulaciones de dedos: `LeftFingerTop/Bot`, `RightFingerTop/Bot`, todas ligadas a su `*Lever_Slider` respectivo.

### Actuadores (`<position>`)
13 actuadores de posición, agrupados por tamaño de servo simulado:

| Familia | Servo ref. | `forcerange` (N·m) | Actuadores |
|---|---|---|---|
| Torso/cuello | DS3218 | ±2.11 (±12.11 en HipBody/BaseHip con gear) | `act_BaseHip`, `act_HipBody`, `act_Neck`, `act_HeadBase`, `act_HeadRot` |
| Brazos/hombros | MG996R | ±2.11 | `act_LeftShoulderArm`, `act_RightShoulderArm`, `act_LeftForearm`, `act_RightForearm` |
| Muñecas | MG90S | ±0.21 | `act_LeftWrist`, `act_RightWrist` |
| Pinza (leva) | MG90S | ±1.08 | `act_LeftLever_Slider`, `act_RightLever_Slider` |

Parámetros que se calibran por actuador: `kp` (proporcional), `kv` (derivativo en velocidad), `gear`, y `damping` del joint asociado.

---

## Reglas de trabajo

### Calibración
- **Métrica principal:** MSE entre `qpos` y target sobre la ventana de simulación.
- **Filtro:** el `qpos` final debe caer dentro de `error_threshold` (típicamente `0.03` rad / m) respecto del target — combinaciones que no lleguen se descartan.
- **Espacio de búsqueda** típico (puede ajustarse por actuador):
  - `kp ∈ arange(2000, 3500, 250)`
  - `kd ∈ arange(0, 60, 20)`
  - `damping ∈ arange(0, 60, 10)`
  - `gear ∈ {1, 5}`
- **Paralelización:** un proceso por actuador via `multiprocessing.Pool(cpu_count()-1)`. Cada worker carga su propia copia del modelo (no compartir `MjModel`/`MjData` entre procesos).
- **Telemetría:** cada worker devuelve `t`, `p` (posición), `f` (fuerza), `target`, y para actuadores con `equality` también los `qpos` de los dedos vinculados (`finger_logs`).
- **act_BaseHip** es problemático — históricamente no encuentra solución en el grid; revisar gear/forcerange/damping si entra al barrido.

### Edición del XML
- Cuando se reexporta desde Fusion360 hay que **reaplicar manualmente** los límites en radianes y el bloque `<equality>` — el plugin no los preserva.
- Mantener nombres del patrón `[Lado][Parte]_[tipo]`; renombrar todo lo que el exportador deje con nombres autogenerados feos.

### Rutas
- `Scripts/calibracion.py` usa `XML_PATH = "Cuerpo\DUM4.xml"` — se corre desde la raíz `DUM_MJC/`, no desde `Scripts/`.
- `Scripts/forzar_motores.ipynb` usa `"..\Cuerpo\DUM4.xml"` — el notebook se corre desde `Scripts/`.

### Comunicación con Claude
- Responder en **español** (rioplatense). Términos técnicos de MuJoCo (`kp`, `damping`, `joint`, etc.) se dejan en inglés porque así están en la API y en el código.
- Antes de tocar parámetros del XML o el grid de búsqueda, leer la sección correspondiente del documento Word más reciente para entender el porqué de los valores actuales.

---

## Comandos útiles

```powershell
# Instalar dependencias
pip install -r Scripts/requirements.txt

# Visualizar el modelo en MuJoCo (binario Windows)
# Arrastrar Cuerpo/DUM4.xml sobre el ejecutable simulate.exe

# Calibración paralela (desde DUM_MJC/)
python Scripts/calibracion.py

# Notebook tutorial
jupyter notebook Scripts/forzar_motores.ipynb
```
