# Control remoto del DUM Pit Droid

Interfaz web que permite comandar el target del robot en tiempo real desde un browser (local o celular en la misma red), mientras la policy entrenada (PPO v13) hace head-tracking de ese target.

## Arquitectura

```
┌──────────────┐       WebSocket        ┌─────────────────────┐
│   Browser    │  ◄──── /ws ────────►   │  FastAPI + uvicorn  │
│ (Canvas 2D)  │                        │ (thread daemon)     │
└──────────────┘                        └──────────┬──────────┘
                                            queue  │ thread-safe
                                                   ▼
                                        ┌─────────────────────┐
                                        │  Sim loop (50 Hz)   │
                                        │  MuJoCo + Policy v13│
                                        │  Viewer interactivo │
                                        └─────────────────────┘
```

- **Cliente** (browser): arrastra un círculo rojo en un canvas 2D que mapea el plano X-Z del world frame.
- **Servidor** (FastAPI): recibe target `{x, y, z}` por WS y lo pone en una `queue.Queue(maxsize=1)`. También envía telemetría `{theta_deg, focus, fps}` al cliente.
- **Sim loop** (main thread): lee target de la cola, calcula la observación, predice acción con la policy, aplica al modelo y sincroniza el viewer.

## Cómo correr

### 1. Dependencias

Ya están en `Scripts/requirements.txt`. Si todavía no las instalaste:

```powershell
& "C:\Users\Elias\AppData\Local\Programs\Python\Python311\python.exe" -m pip install fastapi "uvicorn[standard]" websockets
```

### 2. Arrancar el modo interactivo

```powershell
& "C:\Users\Elias\AppData\Local\Programs\Python\Python311\python.exe" "E:\myspore\Facultad\Mecatronica\ProyectoFinal\DUM_MJC\Scripts\run_interactive.py"
```

Se abre el viewer de MuJoCo + arranca el servidor FastAPI en `http://0.0.0.0:8000`.

### 3. Conectarse desde la PC

Browser → `http://localhost:8000`

Arrastrá el círculo rojo en el pad. El robot reacciona en tiempo real.

### 4. Conectarse desde celular (mismo wifi)

a) Conseguir la IP local de la PC:

```powershell
ipconfig | findstr IPv4
```

Tomar la IP que empieza con `192.168.x.x` o `10.x.x.x`.

b) Abrir el puerto 8000 en el firewall de Windows (una sola vez):

```powershell
# Como administrador:
New-NetFirewallRule -DisplayName "DUM remote" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

c) Desde el celular conectado al mismo wifi: abrir `http://192.168.x.x:8000` en el browser.

## Argumentos opcionales

```
--policy <ruta>            Policy a cargar (default: runs/ppo_dum_v13/final.zip)
--host <ip>                Bind. 0.0.0.0 = LAN, 127.0.0.1 = solo local. Default: 0.0.0.0
--port <int>               Puerto. Default: 8000
--target-smooth <0..1>     LPF sobre el target externo. 1=sin suavizado, 0.3=suave (default).
```

## Cierre

Cerrá el viewer de MuJoCo. El servidor termina al cerrar el proceso Python.

## Troubleshooting

- **El celular no se conecta**: verificá que la PC y el celular estén en la MISMA red wifi, que el firewall tenga la regla creada y que estés usando la IP correcta (no `localhost`).
- **El robot no reacciona al canvas**: el WebSocket puede estar caído. El indicador en el header del HTML dice "conectado" (verde) o "desconectado" (rojo).
- **Lag visible**: revisá el `sim fps` que muestra el HTML. Si está debajo de 40, la PC no llega a 50 Hz; cerrá otras aplicaciones.
- **El viewer no abre**: en Windows a veces requiere `MUJOCO_GL=glfw`. Si falla, exportá `$env:MUJOCO_GL = "glfw"` antes de correr.
