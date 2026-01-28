@echo off
echo ========================================
echo Prompt Enhancer - Environment Check
echo ========================================
echo.

REM Check portable Python
echo [1/4] Checking portable Python...
if exist "portable_python\python.exe" (
    echo ✓ portable_python/python.exe found
    portable_python\python.exe --version
) else (
    echo ✗ portable_python/python.exe NOT found
    echo Please run setup_portable.bat first
    echo.
    pause
    exit /b 1
)
echo.

REM Check dependencies
echo [2/4] Checking dependencies...
portable_python\python.exe -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ✗ FastAPI not installed
    set NEED_DEPS=1
) else (
    echo ✓ FastAPI installed
)

portable_python\python.exe -c "import google.genai" 2>nul
if errorlevel 1 (
    echo ✗ Google Generative AI not installed  
    set NEED_DEPS=1
) else (
    echo ✓ Google Generative AI installed
)

portable_python\python.exe -c "import librosa" 2>nul
if errorlevel 1 (
    echo ✗ Librosa not installed
    set NEED_DEPS=1
) else (
    echo ✓ Librosa installed
)
echo.

REM Check API key
echo [3/4] Checking API key...
if exist ".env" (
    findstr /C:"GOOGLE_API_KEY" .env >nul 2>&1
    if not errorlevel 1 (
        echo ✓ .env file contains GOOGLE_API_KEY
    ) else (
        echo ✗ GOOGLE_API_KEY not found in .env file
        echo Please add your Google API key to .env file
    )
) else (
    echo ✗ .env file not found
    echo Please create .env file with your Google API key
)
echo.

REM Check directories
echo [4/4] Checking directories...
if exist "app\" (
    echo ✓ app/ directory found
) else (
    echo ✗ app/ directory NOT found
)

if exist "uploads\" (
    echo ✓ uploads/ directory found
) else (
    echo ✗ uploads/ directory NOT found (will be created automatically)
)
echo.

REM Summary
echo ========================================
echo SUMMARY:
echo ========================================
if defined NEED_DEPS (
    echo Action needed: Run install_deps.bat to install missing dependencies
) else (
    echo ✓ Everything looks good! You can run run_app.bat
)
echo.

pause
