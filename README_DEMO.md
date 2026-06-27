# Demo del DUM Pit Droid — guía rápida

## Correr la demo (local + remoto, con auto-reinicio)

```powershell
pwsh ./run_demo.ps1
```

Eso levanta el motor de animación con **auto-reinicio** (si el proceso cae, vuelve
solo en ~2 s) y, si tenés `cloudflared` instalado, abre un **túnel** para acceder
desde fuera de casa. La página del cliente **reconecta sola** (WebSocket + video)
cuando el motor vuelve, así una demo remota se recupera sin refrescar el navegador.

- Acceso local: `http://localhost:8000`
- Acceso en la misma red Wi-Fi: `http://<IP-de-tu-PC>:8000`
- Acceso remoto (afuera): la URL `https://...trycloudflare.com` que imprime cloudflared.

## Resiliencia (qué pasa si algo se rompe)

| Problema | Qué lo resuelve |
|---|---|
| El robot queda trabado en un estado raro | Botón **Reiniciar** en la web → reset suave a IDLE, sin cortar nada |
| La física diverge (NaN, el robot "vuela") | El motor lo detecta y hace reset suave automático, sin crashear |
| Una excepción en un step | El motor la atrapa, hace reset suave y sigue |
| El proceso cae del todo (crash duro) | `run_demo.ps1` lo reinicia en ~2 s; el cliente reconecta solo |

El **botón Reiniciar** está siempre visible bajo el estado del robot. Es lo primero
para probar si algo se traba en vivo: no corta la conexión ni reinicia el proceso.

## Acceso remoto — opciones (de más fácil a más técnica)

| Opción | Cómo | Gratis | Notas |
|---|---|---|---|
| **Cloudflare Tunnel** (recomendado) | `cloudflared tunnel --url http://localhost:8000` | Sí | URL `https` estable mientras corre, sin límite de tiempo. Cliente open source |
| **ngrok** | `ngrok http 8000` | Sí (free tier) | El más simple; en el plan free la URL cambia cada vez |
| **localtunnel** | `npx localtunnel --port 8000` | Sí | 100% open source (cliente y server), pero a veces inestable |
| **Tailscale** | VPN privada entre tus dispositivos | Sí | NO es público: solo vos/invitados entran. Más seguro |

### Setup de Cloudflare Tunnel (una sola vez)

1. Descargá `cloudflared` para Windows desde:
   https://github.com/cloudflare/cloudflared/releases
   (el `.exe`; ponelo en el PATH o en la carpeta del proyecto).
2. Listo. `run_demo.ps1` lo detecta y abre el túnel solo.
   O a mano, en otra terminal: `cloudflared tunnel --url http://localhost:8000`

La URL `trycloudflare.com` funciona mientras la PC esté prendida con la demo corriendo.

## Problema típico: el túnel no levanta (DNS)

Si cloudflared falla con algo como:
```
failed to request quick Tunnel: ... lookup api.trycloudflare.com: no such host
```
es que **tu servidor DNS (el del router/ISP) no resuelve el dominio de Cloudflare**.
Se diagnostica así:
```powershell
nslookup api.trycloudflare.com            # tu DNS actual: "Non-existent domain"
nslookup api.trycloudflare.com 1.1.1.1    # via DNS publico: resuelve OK
```
Si lo segundo resuelve y lo primero no, cambiá el DNS de tu PC a uno público.

**Fix (PowerShell como Administrador)** — reemplazá "Ethernet" por tu adaptador
(`Get-NetAdapter` para ver el nombre):
```powershell
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses ("1.1.1.1","8.8.8.8")
Clear-DnsClientCache
```
Para volver al DNS original:
```powershell
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ResetServerAddresses
```
O por la GUI: Configuración → Red e Internet → (tu adaptador) → Editar DNS → Manual →
IPv4 → preferido `1.1.1.1`, alternativo `8.8.8.8`.

## Seguridad

Cualquiera con la URL pública puede controlar el robot. Para una demo está bien.
Si querés que sea **solo tuyo**:
- Usá **Tailscale** (VPN privada) en lugar de un túnel público, o
- Cerrá el túnel (la ventana de cloudflared) cuando no lo uses.

## Correr el motor solo (sin túnel ni auto-reinicio)

```powershell
py -3 Scripts/run_animation_engine.py              # con viewer local
py -3 Scripts/run_animation_engine.py --no-viewer  # headless (solo web)
```
