@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [*] Virtual environment not found. Running setup...
    echo.
    call setup.bat
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Setup failed. Please fix the issues above and try again.
        echo.
        pause
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Virtual environment still missing after setup.
    echo.
    echo Please run setup.bat manually:
    echo   setup.bat
    echo.
    pause
    exit /b 1
)

echo [*] Launching Yavuz...
echo.

.venv\Scripts\python.exe -m yavuz.launcher

endlocal
