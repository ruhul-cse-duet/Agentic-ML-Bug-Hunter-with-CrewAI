@echo off
chcp 65001 >nul 2>&1
cls
echo ================================================================
echo     AGENTIC ML BUG HUNTER - SETUP & INSTALLATION
echo ================================================================
echo.

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.11.14+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/6] Python found!
python --version
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist "venv\" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created!
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo [5/6] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)
echo.

REM Install litellm (required for CrewAI)
echo [6/6] Installing additional dependencies...
pip install litellm
echo.

echo ================================================================
echo                    SETUP COMPLETED!
echo ================================================================
echo.
echo Next steps:
echo   1. Install Ollama: https://ollama.ai/download
echo   2. Start Ollama: ollama serve (in a separate terminal)
echo   3. Pull a model: ollama pull deepseek-coder:1.3b
echo   4. Run the app: run_local.bat
echo.
echo ================================================================
pause
