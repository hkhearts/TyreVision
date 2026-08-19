@echo off
title Tire Vision — AI Inference Server
echo.
echo ============================================================
echo   TIRE VISION — Fleet Tire Intelligence System
echo   TrustGrid Local Inference Server
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

:: Install dependencies
echo [1/3] Installing dependencies...
pip install -r requirements.txt -q

:: Start server
echo [2/3] Starting AI Inference Server on http://localhost:5000
echo [3/3] Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:5000

python server.py
pause
