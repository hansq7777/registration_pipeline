@echo off
setlocal

cd /d "%~dp0"

echo [Histology GUI MVP] Setting up Windows virtual environment...

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found on PATH.
    echo Install Python 3.12+ for Windows and rerun this script.
    pause
    exit /b 1
)

if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    pause
    exit /b 1
)

python -m pip install -r requirements_windows.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo [Histology GUI MVP] Environment is ready.
echo Launch with: start_gui.bat
pause
exit /b 0
