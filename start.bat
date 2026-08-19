@echo off
title Tire Vision - AI Inference Server v2.0
echo.
echo ============================================================
echo   TIRE VISION - Fleet Tire Intelligence System v2.0
echo   TrustGrid Local Inference Server with AI Models
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
echo       Done.

:: Start server
echo [2/3] Starting AI Inference Server on http://localhost:5000
echo       Loading ML models (this may take 10-15 seconds)...
echo [3/3] Opening browser...
timeout /t 3 /nobreak >nul
start http://localhost:5000

python server.py
pause
