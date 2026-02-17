@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Check if venv exists
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [*] Virtual environment not found. Running setup...
    echo.
    call setup.bat
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Setup failed!
        echo.
        pause
        exit /b 1
    )
)

REM Check again after setup
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Virtual environment still missing!
    echo.
    pause
    exit /b 1
)

echo [*] Launching Yavuz...
echo.

uv run python -m yavuz.launcher

