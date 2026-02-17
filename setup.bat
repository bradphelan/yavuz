@echo off
REM Setup script for Yavuz project on Windows
setlocal enabledelayedexpansion

echo.
echo ========================================
echo Yavuz Project Setup
echo ========================================
echo.

REM Check if Python is installed
echo [*] Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python 3.8 or later from one of these sources:
    echo.
    echo   Option 1: Official Python installer (recommended)
    echo   https://www.python.org/downloads/
    echo.
    echo   Option 2: Windows Store
    echo   Search for "Python" in Microsoft Store
    echo.
    echo   Option 3: Chocolatey (if you have it installed)
    echo   choco install python
    echo.
    echo   Option 4: Windows Package Manager (winget)
    echo   winget install Python.Python.3.12
    echo.
    echo Once installed, restart your terminal and run this script again.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] %PYTHON_VERSION% found
echo.

REM Check if pip is installed
echo [*] Checking for pip...
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] pip is not installed!
    echo.
    echo Try running: python -m ensurepip --upgrade
    echo Or reinstall Python with "pip" selected during installation
    echo.
    pause
    exit /b 1
)

echo [OK] pip found
echo.

REM Check if uv is installed
echo [*] Checking for uv...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] uv not found. Installing uv...
    pip install uv
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to install uv
        echo.
        echo Try running this command manually:
        echo   pip install uv
        echo.
        pause
        exit /b 1
    )
    echo [OK] uv installed successfully
) else (
    echo [OK] uv already installed
)
echo.

REM Create virtual environment
echo [*] Setting up virtual environment...
if exist .venv (
    echo [!] Virtual environment already exists at .venv
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        echo [*] Removing old virtual environment...
        rmdir /s /q .venv >nul 2>&1
        echo [OK] Old virtual environment removed
    )
)

if not exist .venv (
    echo [*] Creating new virtual environment...
    uv venv
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment
        echo.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment ready
)
echo.

REM Activate virtual environment
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to activate virtual environment
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Install dependencies
echo [*] Installing dependencies from requirements.txt...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo.
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Install the yavuz package
echo [*] Installing yavuz package in development mode...
uv pip install -e .
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install yavuz package
    echo.
    pause
    exit /b 1
)
echo [OK] Yavuz package installed
echo.

REM Verify installation
echo [*] Verifying installation...
python -c "import yavuz" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] yavuz module could not be imported
    echo.
    echo Try running manually:
    echo   .venv\Scripts\python.exe -m pip install -e .
    echo.
    pause
    exit /b 1
)
echo [OK] yavuz module is importable
echo.

REM Success message
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Your project is ready to use!
echo.
echo Next steps:
echo.
echo   1. Activate the virtual environment:
echo      .venv\Scripts\activate.bat
echo.
echo   2. Launch the demo selector:
echo      start.bat
echo.
echo   3. Or run the launcher directly:
echo      .venv\Scripts\python.exe -m yavuz.launcher
echo.
echo To deactivate the virtual environment later, run:
echo      deactivate
echo.
pause
