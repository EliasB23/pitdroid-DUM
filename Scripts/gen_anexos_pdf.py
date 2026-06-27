"""Genera un PDF por cada archivo de codigo del proyecto, con un header que
explica para que sirvio, mas el codigo con numeros de linea. Salida: Anexos/.

La numeracion de los anexos es AUTOMATICA segun el orden de la lista FILES
(no hay que renumerar a mano si se agrega o quita algo).

Uso:  python Scripts/gen_anexos_pdf.py
"""
from pathlib import Path
import matplotlib
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Anexos"
OUT.mkdir(exist_ok=True)

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
MONO = str(FONT_DIR / "DejaVuSansMono.ttf")
SANS = str(FONT_DIR / "DejaVuSans.ttf")
SANSB = str(FONT_DIR / "DejaVuSans-Bold.ttf")

# (ruta relativa, titulo, para que sirvio).  El numero de anexo se asigna solo
# por la posicion en esta lista.
FILES = [
    # --- MODELO FISICO (MJCF) ---
    ("Cuerpo/DUM4_grab.xml",
     "Modelo fisico final (MJCF)",
     "Modelo del robot en formato MJCF de MuJoCo, version final usada en todo el "
     "desarrollo de agarre y en el motor de animacion. Define los 21 cuerpos rigidos, "
     "las articulaciones con sus limites en radianes, los 15 actuadores de posicion con "
     "sus torques, las restricciones de igualdad leva->dedos, y los elementos de la tarea "
     "de agarre (bola amarilla, sites de palma, equality de sujecion)."),
    ("Cuerpo/DUM4.xml",
     "Modelo fisico base (MJCF)",
     "Modelo MJCF base exportado de Fusion360 (plugin ACDC4Robot) y corregido a mano: "
     "unidades a radianes, limites de articulaciones, propiedades de actuadores y la "
     "restriccion leva->dedos. Es la version sin los elementos de la tarea de agarre, "
     "usada para el entrenamiento del seguimiento con la cabeza."),

    # --- ENTORNOS RL ---
    ("Scripts/rl_env.py",
     "Entorno RL de seguimiento (cabeza)",
     "Entorno Gymnasium DUMHeadTrackingEnv. Define la observacion (18 dim), la accion "
     "(3 joints de la cabeza) y la funcion de recompensa de tracking + foco animatronico "
     "+ estabilidad, mas el shaping que fuerza el uso del giro de cabeza. Base de las "
     "policies v13 y v14c."),
    ("Scripts/rl/envs/grab_env.py",
     "Entorno RL de agarrar y lanzar (brazo)",
     "Entorno Gymnasium DUMGrabEnv. Implementa la maquina de estados FALLING->HELD->THROWN, "
     "el curriculum de dificultad (posicion y caida de la bola) y toda la funcion de "
     "recompensa de catch + hold + throw. Base de las policies v7e a v16."),

    # --- ENTRENAMIENTO ---
    ("Scripts/train_ppo.py",
     "Entrenamiento PPO de la cabeza",
     "Script de entrenamiento de la policy de seguimiento. Incluye el curriculum del cono "
     "de vision (ampliacion del rango de azimuth) y el shaping anti-inclinacion que logro "
     "que la cabeza use el giro (yaw) en vez de inclinar la base (resultado v14c)."),
    ("Scripts/rl/train_grab.py",
     "Entrenamiento PPO del brazo",
     "Script de entrenamiento de la policy de agarre. Incluye los callbacks de curriculum: "
     "ampliacion progresiva del espacio de aparicion de la bola, activacion de la caida, y "
     "rampa de gravedad."),

    # --- EVALUACION ---
    ("Scripts/eval_ppo.py",
     "Evaluacion de la policy de cabeza",
     "Corre N episodios de seguimiento, reporta estadisticas (angulo, foco) y opcionalmente "
     "graba un video con overlay."),
    ("Scripts/eval_v14_head.py",
     "Evaluacion de cabeza con objetivos extremos",
     "Evaluacion especifica con objetivos en azimuth extremo (hasta +-150 grados) para "
     "verificar que la cabeza usa el giro correctamente. Genera video con overlay de los "
     "angulos de cada articulacion."),

    # --- ANIMACION / RUNTIME ---
    ("Scripts/run_animation_engine.py",
     "Motor de animacion integrado (producto final)",
     "Ejecutable final. Combina la policy de cabeza y la del brazo en una sola simulacion, "
     "con maquina de estados (reposo / agarrar / saludar), autofoco procedural, camara "
     "orbital controlable, esquemas de color cambiables y el servidor web. Es lo que se "
     "corre para operar el robot."),
    ("Scripts/run_combined_head_arm.py",
     "Runtime de evaluacion combinado head+brazo",
     "Corre la cabeza y el brazo juntos en una misma simulacion para validar que ambas "
     "policies coexisten, generando video de varias configuraciones de prueba."),
    ("Scripts/run_interactive.py",
     "Runtime interactivo del seguimiento",
     "Version previa solo-cabeza: corre la policy de seguimiento con el objetivo controlado "
     "en tiempo real desde el navegador via WebSocket."),
    ("Scripts/rl/procedural/wave.py",
     "Saludo procedural (sin RL)",
     "Coreografia del saludo escrita como curvas de control en el tiempo (4 fases: retraer "
     "codo, girar muñeca, abrir/cerrar dedos, volver). Es aditiva: escribe los actuadores "
     "del brazo elegido sin interferir con el resto del motor de animacion."),
    ("Scripts/skins.py",
     "Esquemas de color del robot",
     "Define dos apariencias (realista / imperio) y la funcion que las aplica en tiempo real "
     "modificando el color de cada pieza del robot, sin recargar el modelo."),

    # --- WEB ---
    ("Scripts/web_remote/server.py",
     "Servidor web (FastAPI)",
     "Backend de la interfaz. Expone el WebSocket de objetivo/telemetria, el stream de video "
     "MJPEG y los endpoints de acciones (agarrar, saludar, camara, apariencia)."),
    ("Scripts/web_remote/static/index.html",
     "Interfaz cliente (HTML/CSS)",
     "Pagina de control con estetica Star Wars: vista del robot en vivo, pad para mover el "
     "objetivo, y botones de maniobras, camara y apariencia. Pensada para celular en "
     "horizontal."),
    ("Scripts/web_remote/static/control.js",
     "Logica del cliente web (JavaScript)",
     "Captura el arrastre del objetivo en el canvas, dibuja el pad, limita el objetivo al "
     "rango de entrenamiento, envia los comandos y muestra la telemetria recibida."),

    # --- UTILIDADES ---
    ("Scripts/explore_arm_reach.py",
     "Medicion del alcance del brazo",
     "Herramienta interactiva: abre el visor de MuJoCo con sliders y reporta la posicion de "
     "la palma, usada para medir empiricamente el alcance real y definir los limites del "
     "curriculum de agarre."),
    ("Scripts/monitor_v14b.py",
     "Monitor de progreso de entrenamiento",
     "Evalua periodicamente (cada 45 min) el ultimo checkpoint de un entrenamiento de cabeza "
     "en curso, reportando el angulo logrado para varios objetivos de prueba."),
    ("Scripts/calibracion.py",
     "Calibracion inicial de actuadores",
     "Trabajo previo a la etapa de RL: busqueda en grilla paralela (multiprocessing) que "
     "barre kp, kv, damping y gear de cada actuador evaluando estabilidad, sobreimpulso y "
     "precision."),
    ("Scripts/benchmarks.py",
     "Benchmarks iniciales del modelo",
     "Pruebas iniciales (estabilidad, comparacion de integradores, throughput, convergencia "
     "de la calibracion) que sirvieron para decidir el enfoque de aprendizaje por refuerzo."),

    # --- EXPLORATORIOS (no llegaron al pipeline final) ---
    ("Scripts/train_multitask.py",
     "Entrenamiento multi-tarea (exploratorio)",
     "Entrenamiento de un entorno que intentaba combinar cabeza + saludo + pinza en una sola "
     "policy. Linea de trabajo que se reemplazo por policies especializadas separadas. No "
     "formo parte del pipeline final; se incluye por completitud."),
    ("Scripts/eval_multitask.py",
     "Evaluacion multi-tarea (exploratorio)",
     "Evaluacion del entorno multi-tarea exploratorio. No formo parte del pipeline final."),
]


def setup_fonts(pdf):
    pdf.add_font("mono", "", MONO)
    pdf.add_font("sans", "", SANS)
    pdf.add_font("sans", "B", SANSB)


def mcell(pdf, h, txt):
    """multi_cell que SIEMPRE vuelve al margen izquierdo en la linea siguiente."""
    pdf.multi_cell(0, h, txt, new_x="LMARGIN", new_y="NEXT")


def make_index():
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    setup_fonts(pdf)
    pdf.set_auto_page_break(True, margin=14)
    pdf.set_margins(14, 14, 14)
    pdf.add_page()
    pdf.set_font("sans", "B", 16)
    mcell(pdf, 9, "Anexos de codigo — Indice")
    pdf.ln(2)
    pdf.set_font("sans", "", 10)
    mcell(pdf, 5, "Listado de los archivos de codigo del proyecto incluidos como anexos. "
                  "Cada uno tiene su propio documento con el codigo completo y un encabezado "
                  "que explica para que sirvio.")
    pdf.ln(3)
    for num, (rel, titulo, _prop) in enumerate(FILES, 1):
        pdf.set_font("sans", "B", 10)
        mcell(pdf, 5, f"Anexo {num:02d} — {titulo}")
        pdf.set_font("mono", "", 8)
        pdf.set_text_color(90, 90, 90)
        mcell(pdf, 4.5, f"   {rel}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1.5)
    pdf.output(str(OUT / "Anexo_00_indice.pdf"))
    print("OK  Anexo_00_indice.pdf")


def make_file_pdf(num, rel, titulo, prop):
    src_path = ROOT / rel
    code = src_path.read_text(encoding="utf-8", errors="replace").replace("\t", "    ")
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    setup_fonts(pdf)
    pdf.set_auto_page_break(True, margin=12)
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    # --- Header ---
    pdf.set_font("sans", "B", 14)
    mcell(pdf, 7, f"Anexo {num:02d} — {titulo}")
    pdf.ln(0.5)
    pdf.set_font("mono", "", 9)
    pdf.set_text_color(90, 90, 90)
    mcell(pdf, 5, f"Archivo: {rel}")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.set_font("sans", "B", 10)
    mcell(pdf, 5, "Para que sirvio:")
    pdf.set_font("sans", "", 10)
    mcell(pdf, 5, prop)
    pdf.ln(2)
    pdf.set_draw_color(170, 170, 170)
    y = pdf.get_y()
    pdf.line(12, y, 198, y)
    pdf.ln(3)
    # --- Codigo ---
    pdf.set_font("mono", "", 7)
    for i, line in enumerate(code.split("\n"), 1):
        line = line.rstrip("\r")
        mcell(pdf, 3.2, f"{i:>4} | {line}")
    fname = f"Anexo_{num:02d}_{src_path.stem}.pdf"
    pdf.output(str(OUT / fname))
    n_lines = code.count("\n") + 1
    print(f"OK  {fname}  ({n_lines} lineas)")


def main():
    # limpiar PDFs viejos para no dejar numeraciones obsoletas
    for old in OUT.glob("Anexo_*.pdf"):
        old.unlink()
    make_index()
    for num, (rel, titulo, prop) in enumerate(FILES, 1):
        make_file_pdf(num, rel, titulo, prop)
    print(f"\nListo. {len(FILES)+1} PDFs en {OUT}")


if __name__ == "__main__":
    main()
