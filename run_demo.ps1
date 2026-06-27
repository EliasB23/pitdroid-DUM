# ============================================================
#  Lanzador de demo  -  DUM Pit Droid
#  - Motor de animacion con AUTO-REINICIO: si el proceso cae, vuelve solo.
#  - Tunel Cloudflare para acceso remoto (si 'cloudflared' esta instalado).
#  La pagina del cliente reconecta sola (WebSocket + video) cuando el motor vuelve,
#  asi una demo remota se recupera sin tener que refrescar el navegador.
#
#  Uso:   pwsh ./run_demo.ps1       (o boton derecho -> Ejecutar con PowerShell)
#  Salir: Ctrl+C
# ============================================================
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot          # correr siempre desde la raiz del proyecto
$Port = 8000
$Engine = Join-Path $PSScriptRoot "Scripts\run_animation_engine.py"

# cloudflared se busca, en orden: variable de entorno CLOUDFLARED ->
# PATH del sistema -> el .exe dentro de la carpeta del repo. No se hardcodea
# ninguna ruta del filesystem: si lo tenes en otro lado, exporta $env:CLOUDFLARED
# o copia el .exe a esta carpeta. (Ver README_DEMO.md)
$CloudflaredPath = $env:CLOUDFLARED

function Find-Cloudflared {
    if ($CloudflaredPath -and (Test-Path $CloudflaredPath)) { return $CloudflaredPath }
    $g = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($g) { return $g.Source }
    foreach ($p in @(
        (Join-Path $PSScriptRoot "cloudflared.exe"),
        (Join-Path $PSScriptRoot "cloudflared-windows-amd64.exe")
    )) { if (Test-Path $p) { return $p } }
    return $null
}

Write-Host ""
Write-Host "=== DUM Pit Droid - Demo ===" -ForegroundColor Yellow

# 1) Tunel Cloudflare (acceso desde fuera de casa)
$cfProc = $null
$cfExe = Find-Cloudflared
if ($cfExe) {
    $cfOut = Join-Path $PSScriptRoot "cloudflared.out.log"
    $cfErr = Join-Path $PSScriptRoot "cloudflared.err.log"
    Remove-Item $cfOut, $cfErr -ErrorAction SilentlyContinue
    Write-Host "[demo] Iniciando tunel Cloudflare ($cfExe)..." -ForegroundColor Cyan
    $cfProc = Start-Process $cfExe -ArgumentList "tunnel","--url","http://localhost:$Port" `
        -RedirectStandardOutput $cfOut -RedirectStandardError $cfErr -NoNewWindow -PassThru
    # Esperar a leer la URL publica (o detectar que fallo)
    $url = $null
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 700
        $txt = ""
        if (Test-Path $cfErr) { $txt += (Get-Content $cfErr -Raw -ErrorAction SilentlyContinue) }
        if (Test-Path $cfOut) { $txt += (Get-Content $cfOut -Raw -ErrorAction SilentlyContinue) }
        # Detectar el fallo PRIMERO (api.trycloudflare.com aparece en la linea de error)
        if ($txt -match "no such host|failed to request quick Tunnel") { break }
        # La URL real es de palabras random; excluimos api.trycloudflare.com (endpoint interno)
        $m = [regex]::Match($txt, "https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com")
        if ($m.Success) { $url = $m.Value; break }
        if ($cfProc.HasExited) { break }
    }
    if ($url) {
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "  URL PUBLICA (abrila desde el celu / fuera de casa):" -ForegroundColor Green
        Write-Host "    $url" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
    } else {
        Write-Host "[demo] El tunel NO levanto. Revisa: $cfErr" -ForegroundColor Red
        Write-Host "[demo] Causa tipica: el DNS no resuelve cloudflare." -ForegroundColor Red
        Write-Host "[demo] Fix: cambia el DNS a 1.1.1.1 (ver README_DEMO.md, seccion DNS)." -ForegroundColor Red
    }
} else {
    Write-Host "[demo] cloudflared NO encontrado -> SOLO acceso local:" -ForegroundColor Red
    Write-Host "       http://localhost:$Port   (o http://<IP-de-esta-PC>:$Port en la misma red Wi-Fi)" -ForegroundColor Red
    Write-Host "       Pone cloudflared.exe en esta carpeta, en el PATH, o exporta `$env:CLOUDFLARED." -ForegroundColor Red
}

Write-Host ""
Write-Host "[demo] Acceso local:  http://localhost:$Port" -ForegroundColor Green
if ($url) {
    Write-Host "[demo] IMPORTANTE: abri la URL publica RECIEN cuando veas abajo" -ForegroundColor Yellow
    Write-Host "       '[runtime] estado inicial: IDLE' (el motor tarda ~10s en cargar)." -ForegroundColor Yellow
    Write-Host "       Si dice 'not found', el motor todavia no termino de levantar: refresca." -ForegroundColor Yellow
}
Write-Host "[demo] Motor con auto-reinicio. Ctrl+C para terminar todo." -ForegroundColor Yellow
Write-Host ""

# 2) Motor con auto-reinicio
try {
    while ($true) {
        py -3 $Engine --no-viewer
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            Write-Host "[demo] El motor termino normalmente. Saliendo." -ForegroundColor Yellow
            break
        }
        Write-Host "[demo] El motor cayo (codigo $code). Reiniciando en 2s..." -ForegroundColor Red
        Start-Sleep -Seconds 2
    }
}
finally {
    if ($cfProc -and -not $cfProc.HasExited) {
        Write-Host "[demo] Cerrando tunel Cloudflare..." -ForegroundColor Cyan
        Stop-Process -Id $cfProc.Id -ErrorAction SilentlyContinue
    }
}
