@echo off
chcp 65001 >nul 2>&1
cls
echo ================================================================
echo         AGENTIC ML BUG HUNTER - LOCAL LAUNCH
echo ================================================================
echo.

REM Color definitions
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set BLUE=[94m
set RESET=[0m

echo %BLUE%[2/6]%RESET% Checking Ollama service...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%[WARNING]%RESET% Ollama is not running!
    echo.
    echo Please start Ollama in a separate terminal:
    echo   1. Open new terminal/PowerShell
    echo   2. Run: ollama serve
    echo   3. Keep it running
    echo.
    echo Then press any key to continue...
    pause >nul

    REM Check again
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if %errorlevel% neq 0 (
        echo %RED%[ERROR]%RESET% Ollama is still not running!
        echo Please start Ollama and try again.
        pause
        exit /b 1
    )
)
echo %GREEN%[SUCCESS]%RESET% Ollama is running
echo.

echo %BLUE%[3/6]%RESET% Checking available models...
ollama list | findstr /i "ollama/deepseek-coder:1.3b" >nul 2>&1
if %errorlevel% neq 0 (
    echo %YELLOW%[WARNING]%RESET% deepseek-coder:1.3b or gemma2:2b not found!
    echo.
    echo Available models:
    ollama list
    echo.
    set /p PULL_MODEL="Do you want to pull deepseek-coder:1.3b? (y/n): "
    if /i "%PULL_MODEL%"=="y" (
        echo.
        echo Pulling deepseek-coder:1.3b... (This may take several minutes)
        ollama pull deepseek-coder:1.3b
        echo.
        echo Model pull completed. Continuing...
    ) else (
        echo.
        echo Using already pulled model.
    )

)
echo %GREEN%[SUCCESS]%RESET% Model available
echo.

echo %BLUE%[4/6]%RESET% Checking .env configuration...
if not exist ".env" (
    echo %RED%[ERROR]%RESET% .env file not found!
    echo Please ensure .env file exists in the project root.
    pause
    exit /b 1
)
echo %GREEN%[SUCCESS]%RESET% Configuration file found
echo.

echo %BLUE%[5/6]%RESET% Checking required files...
set FILES_OK=1
if not exist "app\main.py" set FILES_OK=0
if not exist "app\crew_runner.py" set FILES_OK=0
if not exist "agents\runtime_agent.py" set FILES_OK=0
if not exist "llm_model.py" set FILES_OK=0
if not exist "config.py" set FILES_OK=0

if %FILES_OK% equ 0 (
    echo %RED%[ERROR]%RESET% Some project files are missing!
    pause
    exit /b 1
)
echo %GREEN%[SUCCESS]%RESET% All required files present
echo.

echo %BLUE%[6/6]%RESET% Starting application...
echo.
echo ================================================================
echo   Application starting at: http://127.0.0.1:8000
echo   Press CTRL+C to stop the server
echo ================================================================
echo.
echo %GREEN%[INFO]%RESET% Using Ollama Model: deepseek-coder:1.3b
echo %GREEN%[INFO]%RESET% Temperature: 0.4
echo %GREEN%[INFO]%RESET% Max Tokens: 512
echo.
echo ================================================================
echo.

cd app
python main.py

if %errorlevel% neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Application failed to start!
    echo.
    echo Common issues:
    echo   1. Port 8000 already in use
    echo   2. Missing dependencies (run: pip install -r requirements.txt)
    echo   3. Ollama not properly configured
    echo.
    pause
)
