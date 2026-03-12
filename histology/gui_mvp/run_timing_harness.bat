@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD="
set "LOG_DIR=%~dp0logs"
set "LOG_FILE=%LOG_DIR%\timing_harness_%RANDOM%.log"

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
set "PYTHONPATH=%~dp0"
"%PYTHON_CMD%" "%~dp0timing_harness_windows.py" %* 1>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo.
    echo Timing harness failed.
    echo See log: %LOG_FILE%
    type "%LOG_FILE%"
    exit /b 1
)
type "%LOG_FILE%"
exit /b 0

:no_python
echo Python not found. Run setup_windows_env.bat first.
exit /b 1
