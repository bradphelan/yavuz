@echo off
REM Setup script for Yavuz project on Windows

echo ========================================
echo Yavuz Project Setup
echo ========================================
echo.

REM Check if uv is installed
echo Checking for uv...
uv --version > nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    pip install uv
    if %errorlevel% neq 0 (
        echo Failed to install uv. Please install it manually: pip install uv
        exit /b 1
    )
)

echo uv is installed.
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists.
) else (
    uv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment.
        exit /b 1
    )
)

echo.
echo Installing dependencies...
call .venv\Scripts\activate.bat
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    exit /b 1
)

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo     .venv\Scripts\activate.bat
echo.
echo To launch the demo selector, run:
echo     python launcher.py
echo.
echo Or run the launcher directly:
echo     .venv\Scripts\python.exe launcher.py
echo.
pause
