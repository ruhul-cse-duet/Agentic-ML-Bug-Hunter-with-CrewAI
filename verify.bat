@echo off
chcp 65001 >nul 2>&1
cls
echo ================================================================
echo         AGENTIC ML BUG HUNTER - SYSTEM VERIFICATION
echo ================================================================
echo.

set ERROR_COUNT=0
set SUCCESS_COUNT=0

REM Check 1: Python Installation
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Python is installed
    python --version
    set /a SUCCESS_COUNT+=1
) else (
    echo [ERROR] Python is NOT installed!
    echo Please install Python 3.11.14+ from: https://www.python.org/downloads/
    set /a ERROR_COUNT+=1
)
echo.

REM Check 2: Virtual Environment
echo [2/8] Checking virtual environment...
if exist "venv\" (
    echo [SUCCESS] Virtual environment exists
    set /a SUCCESS_COUNT+=1
) else (
    echo [ERROR] Virtual environment NOT found!
    echo Please run: setup.bat
    set /a ERROR_COUNT+=1
)
echo.

REM Check 3: Ollama Installation
echo [3/8] Checking Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is installed
    ollama --version
    set /a SUCCESS_COUNT+=1
) else (
    echo [ERROR] Ollama is NOT installed!
    echo Please install from: https://ollama.ai/download
    set /a ERROR_COUNT+=1
)
echo.

REM Check 4: Ollama Service
echo [4/8] Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Ollama is running
    set /a SUCCESS_COUNT+=1
) else (
    echo [ERROR] Ollama is NOT running!
    echo Please run: ollama serve
    set /a ERROR_COUNT+=1
)
echo.

REM Check 5: Model Availability
echo [5/8] Checking if models are available...
ollama list | findstr /i "deepseek-coder:1.3b" >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Compatible model found
    ollama list | findstr /i "deepseek-coder:1.3b"
    set /a SUCCESS_COUNT+=1
) else (
    echo [ERROR] No compatible model found!
    echo Please run: ollama pull deepseek-coder:1.3b
    set /a ERROR_COUNT+=1
)
echo.

REM Check 6: Dependencies (if venv exists)
if exist "venv\" (
    echo [6/8] Checking Python dependencies...
    call venv\Scripts\activate.bat
    pip show fastapi >nul 2>&1
    if %errorlevel% equ 0 (
        echo [SUCCESS] Dependencies are installed
        set /a SUCCESS_COUNT+=1
    ) else (
        echo [ERROR] Dependencies NOT installed!
        echo Please run: setup.bat
        set /a ERROR_COUNT+=1
    )
    echo.
) else (
    echo [6/8] Skipping dependency check (no venv)
    echo.
)

REM Check 7: Port Availability
echo [7/8] Checking if port 8000 is available...
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARNING] Port 8000 is already in use!
    echo You may need to stop other services or use a different port.
) else (
    echo [SUCCESS] Port 8000 is available
    set /a SUCCESS_COUNT+=1
)
echo.

REM Check 8: Required Files
echo [8/8] Checking project files...
set FILES_OK=1
if not exist "app\main.py" set FILES_OK=0
if not exist "app\crew_runner.py" set FILES_OK=0
if not exist "agents\runtime_agent.py" set FILES_OK=0
if not exist "llm_model.py" set FILES_OK=0
if not exist "config.py" set FILES_OK=0
if not exist ".env" set FILES_OK=0

if %FILES_OK% equ 1 (
    echo [SUCCESS] All required files present
    set /a SUCCESS_COUNT+=1
) else (
    echo [ERROR] Some project files are missing!
    set /a ERROR_COUNT+=1
)
echo.

REM Summary
echo ================================================================
echo                       VERIFICATION SUMMARY
echo ================================================================
echo.
echo Total Checks: 8
echo Passed: %SUCCESS_COUNT%
echo Failed: %ERROR_COUNT%
echo.

if %ERROR_COUNT% equ 0 (
    echo ================================================================
    echo              ALL CHECKS PASSED! YOU'RE READY! 
    echo ================================================================
    echo.
    echo Next Steps:
    echo   1. Run: run_local.bat
    echo   2. Open: http://localhost:8000
    echo   3. Start debugging ML code!
    echo.
    echo ================================================================
) else (
    echo ================================================================
    echo            SOME CHECKS FAILED - PLEASE FIX ABOVE
    echo ================================================================
    echo.
    echo Quick Fixes:
    echo   - Missing Python? Install from python.org
    echo   - Missing Ollama? Install from ollama.ai
    echo   - No venv? Run: setup.bat
    echo   - Ollama not running? Run: ollama serve
    echo   - Model missing? Run: ollama pull llama2:7b
    echo.
    echo ================================================================
)

pause
