@echo off
echo ========================================
echo Starting Prompt Enhancer
echo ========================================
echo.

REM Check if portable Python exists, if not run automatic setup
if not exist "portable_python\python.exe" (
    echo Portable Python not found. Running automatic setup...
    echo.
    call setup_portable.bat
    if errorlevel 1 (
        echo Setup failed. Please try again manually.
        pause
        exit /b 1
    )
)

REM Check if dependencies are installed
portable_python\python.exe -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Dependencies not found. Installing now...
    call install_deps.bat
    if errorlevel 1 (
        echo Dependency installation failed.
        pause
        exit /b 1
    )
)

echo Starting FastAPI server...
echo.
echo The app will be available at: http://localhost:8002
echo Press Ctrl+C to stop the server
echo.

REM Open browser after a short delay
start /min timeout /t 3 /nobreak >nul 2>&1
start http://localhost:8002

REM Start the FastAPI app
portable_python\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

pause
