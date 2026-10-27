@echo off
title SolarGuard AI Dashboard
echo ===================================================
echo Starting SolarGuard AI Dashboard...
echo ===================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment exists and activate if present
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
)

echo Launching Streamlit dashboard...
echo The dashboard will automatically open in your default browser at http://localhost:8501
echo (Press Ctrl+C to stop the server)
echo.

python -m streamlit run app.py

pause
