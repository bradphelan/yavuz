@echo off
REM Setup script for Yavuz project on Windows

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Yavuz Project Setup
echo ========================================
echo.

REM Check if Python is installed (try py first)
echo [*] Checking for Python...
py --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo [OK] !PYTHON_VERSION!
    set PYTHON_CMD=py
) else (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
        echo [OK] !PYTHON_VERSION!
        set PYTHON_CMD=python
    ) else (
        echo [ERROR] Python is not installed!
        echo.
        echo Please install Python from: https://www.python.org/downloads/
        echo.
        pause
        exit /b 1
    )
)
echo.

REM Check if uv is installed
echo [*] Checking for uv...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Installing uv...
    !PYTHON_CMD! -m pip install uv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install uv
        echo.
        pause
        exit /b 1
    )
)
echo [OK] uv is installed
echo.

REM Run uv sync to create venv and install everything
echo [*] Running uv sync...
uv sync
if %errorlevel% neq 0 (
    echo [ERROR] uv sync failed
    echo.
    pause
    exit /b 1
)
echo.

REM Success
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Your project is ready to use!
echo.
echo To launch the app:
echo   start.bat
echo   or
echo   python -m yavuz.launcher
echo.
pause
