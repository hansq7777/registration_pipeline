@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="
set "LOG_DIR=%~dp0logs"
set "LOG_FILE=%LOG_DIR%\start_gui_%RANDOM%.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "usebackq delims=" %%I in (`where python`) do (
        echo %%I | findstr /I "ProgramData\\anaconda3\\python.exe" >nul
        if not errorlevel 1 (
            set "PYTHON_CMD=%%I"
            goto launch
        )
    )
    if "%PYTHON_CMD%"=="" set "PYTHON_CMD=python"
)

if "%PYTHON_CMD%"=="" if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
)

if "%PYTHON_CMD%"=="" goto no_python

:launch
if /I "%~1"=="--check" (
    echo [Histology GUI MVP] Python command: %PYTHON_CMD%
    "%PYTHON_CMD%" --version
    exit /b %ERRORLEVEL%
)

echo [Histology GUI MVP] Starting GUI...
set "PYTHONPATH=%~dp0"
"%PYTHON_CMD%" "%~dp0self_check.py" 1>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    if /I not "%PYTHON_CMD%"=="%~dp0.venv\Scripts\python.exe" if exist "%~dp0.venv\Scripts\python.exe" (
        echo [Histology GUI MVP] Retrying with local .venv...
        set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
        "%PYTHON_CMD%" "%~dp0self_check.py" 1>"%LOG_FILE%" 2>&1
    )
)
if errorlevel 1 (
    echo.
    echo GUI failed to start.
    echo See log: %LOG_FILE%
    echo If this is the first launch, run setup_windows_env.bat
    pause
    type "%LOG_FILE%"
)
if not errorlevel 1 (
    start "Histology GUI MVP" "%PYTHON_CMD%" "%~dp0bootstrap_qt_env.py"
)

exit /b 0

:no_python
echo Python not found. Run setup_windows_env.bat first.
pause
exit /b 1
