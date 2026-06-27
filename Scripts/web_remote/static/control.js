// Control remoto del target del DUM Pit Droid.
// Canvas representa el plano (X lateral, Z vertical) en world frame.
// Cliente envia {x, y, z} cada vez que el usuario mueve el target.
// Servidor envia telemetria {theta_deg, focus, fps}.

const PAD = document.getElementById("pad");
const CTX = PAD.getContext("2d");
const STATUS = document.getElementById("status");
const DIST_SLIDER = document.getElementById("dist-slider");
const DIST_VAL = document.getElementById("dist-val");
const THETA_VAL = document.getElementById("theta-val");
const FOCUS_VAL = document.getElementById("focus-val");
const FOCUS_CELL = document.getElementById("focus-cell");
const FPS_VAL = document.getElementById("fps-val");
const STATE_VAL = document.getElementById("state-val");
const GRAB_BTN = document.getElementById("grab-btn");
const WAVE_BTN = document.getElementById("wave-btn");
const SKIN_REALISTA = document.getElementById("skin-realista");
const SKIN_IMPERIO = document.getElementById("skin-imperio");

// Rango de coordenadas WORLD que cubre el pad. Calibrado al modelo real:
// posicion del lente en pose neutra (medida con mujoco.site_xpos['lens_center']).
// - X world: lateral. Pad horizontal -> X world.
// - Z world: vertical. Pad vertical -> Z world (invertido para que arriba=+Z).
// - Y world: distancia frontal/trasera. Controlada por el slider con signo.
//   El robot mira hacia -Y world en pose neutra. Distancia positiva = al frente (-Y world);
//   negativa = detras (+Y world). El slider va de -1 (atras) a +1 (adelante).
const LENS_POS_WORLD = { x: 0.115, y: 0.139, z: 0.503 };
const PAD_RANGE_X = 1.2;  // metros (ancho del pad en world X). +/- 0.6 a cada lado del lente.
const PAD_RANGE_Z = 1.0;  // metros (alto del pad en world Z). +/- 0.5 (arriba/abajo).
// v14c training distancia 0.3..1.0m. Limito el slider para no salirse del rango entrenado.
const DIST_MIN = -0.65;   // backward dentro del cono ±150° azimuth y dist <= 0.7m
const DIST_MAX =  0.95;   // frente, max 0.95m (training: 1.0m max, dejo margen)
// Clamp final del target para que la distancia 3D al lens este en [TRAIN_DIST_MIN, TRAIN_DIST_MAX]
const TRAIN_DIST_MIN = 0.30;
const TRAIN_DIST_MAX = 1.00;

// Estado del target en world coords
let target = {
  x: LENS_POS_WORLD.x,
  y: LENS_POS_WORLD.y - 0.6,   // arranca 0.6m al frente
  z: LENS_POS_WORLD.z,
};
let signedDist = 0.6;  // -0.8 .. +1.0

let dragging = false;

// --- canvas drawing ---
function drawPad() {
  const w = PAD.width, h = PAD.height;
  // bg
  CTX.fillStyle = "#2a2a30";
  CTX.fillRect(0, 0, w, h);

  // grid
  CTX.strokeStyle = "#3a3a45";
  CTX.lineWidth = 1;
  for (let i = 1; i < 8; i++) {
    const x = (w * i) / 8;
    const y = (h * i) / 8;
    CTX.beginPath();
    CTX.moveTo(x, 0); CTX.lineTo(x, h);
    CTX.moveTo(0, y); CTX.lineTo(w, y);
    CTX.stroke();
  }

  // ejes centrales
  CTX.strokeStyle = "#555";
  CTX.lineWidth = 1.5;
  CTX.beginPath();
  CTX.moveTo(w / 2, 0); CTX.lineTo(w / 2, h);
  CTX.moveTo(0, h / 2); CTX.lineTo(w, h / 2);
  CTX.stroke();

  // etiquetas
  CTX.fillStyle = "#999";
  CTX.font = "12px system-ui";
  CTX.fillText("← izquierda", 6, h / 2 - 6);
  CTX.fillText("derecha →", w - 70, h / 2 - 6);
  CTX.fillText("↑ arriba", w / 2 + 6, 14);
  CTX.fillText("↓ abajo", w / 2 + 6, h - 8);

  // target
  const px = ((target.x - LENS_POS_WORLD.x) / PAD_RANGE_X + 0.5) * w;
  const py = ((LENS_POS_WORLD.z - target.z) / PAD_RANGE_Z + 0.5) * h; // Z arriba = py chico
  CTX.beginPath();
  CTX.arc(px, py, 18, 0, Math.PI * 2);
  CTX.fillStyle = "#ff4747";
  CTX.fill();
  CTX.strokeStyle = "#fff";
  CTX.lineWidth = 2;
  CTX.stroke();
}

// --- pad -> world ---
function padToWorld(px, py) {
  const w = PAD.width, h = PAD.height;
  const x = LENS_POS_WORLD.x + (px / w - 0.5) * PAD_RANGE_X;
  const z = LENS_POS_WORLD.z - (py / h - 0.5) * PAD_RANGE_Z;
  return { x, z };
}

// --- WebSocket ---
let ws;
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => {
    STATUS.textContent = "conectado";
    STATUS.className = "ok";
    sendTarget();  // envia el target inicial
  };
  ws.onclose = () => {
    STATUS.textContent = "desconectado — reintentando…";
    STATUS.className = "err";
    setTimeout(connect, 1500);
  };
  ws.onerror = () => {
    STATUS.textContent = "error";
    STATUS.className = "err";
  };
  ws.onmessage = (e) => {
    try {
      const t = JSON.parse(e.data);
      if (typeof t.theta_deg === "number") {
        THETA_VAL.textContent = t.theta_deg.toFixed(1) + " °";
      }
      if (typeof t.focus === "boolean") {
        FOCUS_VAL.textContent = t.focus ? "YES" : "no";
        FOCUS_VAL.className = "val " + (t.focus ? "YES" : "no");
      }
      if (typeof t.fps === "number") {
        FPS_VAL.textContent = t.fps.toFixed(0);
      }
      if (typeof t.state === "string" && STATE_VAL) {
        STATE_VAL.textContent = t.state;
        STATE_VAL.style.color = (t.state === "IDLE") ? "var(--mute)" : "var(--ok)";
        // Actualizar habilitacion de botones segun el SM real (no por timeout)
        if (t.state !== currentSMState) {
          currentSMState = t.state;
          updateButtonStateFromSM();
        }
      }
    } catch (_) {}
  };
}

// Estado actual del SM segun telemetria (se actualiza desde ws.onmessage)
let currentSMState = "IDLE";

function updateButtonStateFromSM() {
  // Los botones de accion solo se habilitan en IDLE
  const inIdle = (currentSMState === "IDLE");
  if (GRAB_BTN) {
    GRAB_BTN.disabled = !inIdle;
    if (inIdle) GRAB_BTN.textContent = "AGARRAR BOLA";
    else GRAB_BTN.textContent = "AGARRAR (esperá: " + currentSMState + ")";
  }
  if (WAVE_BTN) {
    WAVE_BTN.disabled = !inIdle;
    if (inIdle) WAVE_BTN.textContent = "SALUDAR (procedural 6s)";
    else WAVE_BTN.textContent = "SALUDAR (esperá: " + currentSMState + ")";
  }
}

if (GRAB_BTN) {
  GRAB_BTN.addEventListener("click", async () => {
    if (currentSMState !== "IDLE") return;
    GRAB_BTN.disabled = true;
    GRAB_BTN.textContent = "...";
    try {
      await fetch("/grab_yellow", { method: "POST" });
    } catch (e) {
      GRAB_BTN.textContent = "ERROR de red";
    }
    // No timeout: la habilitacion vuelve cuando la telemetria reporta IDLE
  });
}

if (WAVE_BTN) {
  WAVE_BTN.addEventListener("click", async () => {
    if (currentSMState !== "IDLE") return;
    WAVE_BTN.disabled = true;
    WAVE_BTN.textContent = "...";
    try {
      await fetch("/wave", { method: "POST" });
    } catch (e) {
      WAVE_BTN.textContent = "ERROR de red";
    }
  });
}

// Botones de apariencia (skin) — funcionan en cualquier estado, son cosméticos
async function setSkin(name) {
  try { await fetch("/skin/" + name, { method: "POST" }); } catch (e) {}
  // Marcar el botón activo
  if (SKIN_REALISTA) SKIN_REALISTA.classList.toggle("skin-active", name === "realista");
  if (SKIN_IMPERIO)  SKIN_IMPERIO.classList.toggle("skin-active", name === "imperio");
  // Cambiar el TEMA de la página para que matchee con la skin del robot
  document.body.classList.toggle("theme-imperial", name === "imperio");
}
if (SKIN_REALISTA) SKIN_REALISTA.addEventListener("click", () => setSkin("realista"));
if (SKIN_IMPERIO)  SKIN_IMPERIO.addEventListener("click", () => setSkin("imperio"));

// D-pad de cámara: izq/der giran, arriba/abajo cambian altura (cosmético)
async function moveCam(dir) {
  try { await fetch("/cam/" + dir, { method: "POST" }); } catch (e) {}
}
["up", "down", "left", "right"].forEach((dir) => {
  const btn = document.getElementById("cam-" + dir);
  if (btn) btn.addEventListener("click", () => moveCam(dir));
});

// Botón Reiniciar: reset suave del robot (no corta la conexión)
const RESET_BTN = document.getElementById("reset-btn");
if (RESET_BTN) {
  RESET_BTN.addEventListener("click", async () => {
    RESET_BTN.disabled = true;
    const prev = RESET_BTN.textContent;
    RESET_BTN.textContent = "...";
    try { await fetch("/reset", { method: "POST" }); } catch (e) {}
    setTimeout(() => { RESET_BTN.disabled = false; RESET_BTN.textContent = prev; }, 1200);
  });
}

// --- Reconexión automática del video MJPEG ---
// Si el backend se reinicia, el <img src="/stream"> se rompe. Lo recargamos solo
// (con cache-buster) para que la demo remota se recupere sin refrescar la página.
const ROBOT_VIEW = document.getElementById("robot-view");
if (ROBOT_VIEW) {
  ROBOT_VIEW.addEventListener("error", () => {
    setTimeout(() => {
      ROBOT_VIEW.src = "/stream?t=" + Date.now();
    }, 1500);
  });
}

function clampTargetToTrainingDist(t) {
  // Asegura que la distancia 3D del target al lens este en [TRAIN_DIST_MIN, TRAIN_DIST_MAX].
  // Si esta fuera, se mueve a lo largo del rayo lens->target hasta caer en rango.
  const dx = t.x - LENS_POS_WORLD.x;
  const dy = t.y - LENS_POS_WORLD.y;
  const dz = t.z - LENS_POS_WORLD.z;
  const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
  if (d < 1e-6) return t;  // protege contra div/0
  let factor = 1.0;
  if (d > TRAIN_DIST_MAX) factor = TRAIN_DIST_MAX / d;
  else if (d < TRAIN_DIST_MIN) factor = TRAIN_DIST_MIN / d;
  return {
    x: LENS_POS_WORLD.x + dx * factor,
    y: LENS_POS_WORLD.y + dy * factor,
    z: LENS_POS_WORLD.z + dz * factor,
  };
}

function sendTarget() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    const clamped = clampTargetToTrainingDist(target);
    ws.send(JSON.stringify(clamped));
  }
}

// --- input events ---
function updateFromPointer(ev) {
  const rect = PAD.getBoundingClientRect();
  const ptr = ev.touches ? ev.touches[0] : ev;
  const px = ((ptr.clientX - rect.left) / rect.width) * PAD.width;
  const py = ((ptr.clientY - rect.top) / rect.height) * PAD.height;
  const w = padToWorld(px, py);
  target.x = w.x;
  target.z = w.z;
  drawPad();
  sendTarget();
}

PAD.addEventListener("pointerdown", (e) => { dragging = true; PAD.setPointerCapture(e.pointerId); updateFromPointer(e); });
PAD.addEventListener("pointermove", (e) => { if (dragging) updateFromPointer(e); });
PAD.addEventListener("pointerup",   (e) => { dragging = false; });
PAD.addEventListener("pointercancel",(e) => { dragging = false; });

function applyDist() {
  // d signed: positivo = adelante (-Y world); negativo = atras (+Y world).
  signedDist = parseFloat(DIST_SLIDER.value) / 100;
  if (signedDist >= 0) {
    DIST_VAL.textContent = signedDist.toFixed(2) + " m frente";
  } else {
    DIST_VAL.textContent = Math.abs(signedDist).toFixed(2) + " m atras";
  }
  // target.y = lente.y − dist_signed (frente = -Y, atras = +Y)
  target.y = LENS_POS_WORLD.y - signedDist;
  sendTarget();
}
DIST_SLIDER.addEventListener("input", applyDist);

// Init
drawPad();
connect();
