@echo off
echo ========================================
echo Prompt Enhancer - Installing Dependencies
echo ========================================
echo.

REM Check if portable Python exists
if not exist "portable_python\python.exe" (
    echo ERROR: portable_python directory not found!
    echo Please download Python embeddable package from:
    echo https://www.python.org/downloads/windows/
    echo Extract it as 'portable_python' folder in this directory
    echo.
    pause
    exit /b 1
)

echo Found portable Python, installing dependencies...
echo.

REM Upgrade pip first
portable_python\python.exe -m pip install --upgrade pip

REM Install dependencies
portable_python\python.exe -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo You can now run the app with run_app.bat
echo.
pause
