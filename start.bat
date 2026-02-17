@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Running setup...
    call setup.bat
)

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment still missing. Run setup.bat and try again.
    exit /b 1
)

.venv\Scripts\python.exe -m yavuz.launcher

endlocal
