# SolarGuard AI - PowerShell Launch Script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting SolarGuard AI Dashboard..." -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if present
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (.venv)..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment (venv)..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

Write-Host "Launching Streamlit dashboard..." -ForegroundColor Green
Write-Host "The dashboard will automatically open at http://localhost:8501" -ForegroundColor Green
Write-Host "(Press Ctrl+C to stop the server)" -ForegroundColor Gray
Write-Host ""

python -m streamlit run app.py
