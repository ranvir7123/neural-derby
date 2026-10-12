# Start the Neural Derby show on this laptop.
# Makes a fresh certificate for the laptop's current network address, then starts the server.
# Run it from anywhere:  & "C:\dev\study\Edge-Ai\.claude\worktrees\elastic-tesla-5be637\neural-derby\tools\start_show.ps1"
# Stop the server with Ctrl-C in this window.

$root = Split-Path -Parent $PSScriptRoot      # the neural-derby folder
$python = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

Write-Host "1/2  Making the certificate for this network..." -ForegroundColor Cyan
& $python tools\make_cert.py
if ($LASTEXITCODE -ne 0) { Write-Host "Certificate failed. Is the laptop on a network?" -ForegroundColor Red; exit 1 }

Write-Host "2/2  Starting the server (Ctrl-C stops it)..." -ForegroundColor Cyan
Write-Host "     Projector page : http://localhost:8000/stadium/"
Write-Host "     Control page   : http://localhost:8000/control/   (the phone links are shown there)"
& $python -m derby.server.main
