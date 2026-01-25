@echo off
echo ========================================
echo Prompt Enhancer - Installing Dependencies
echo ========================================
echo.

REM Check if portable Python exists
if not exist "portable_python\python.exe" (
    echo ERROR: portable_python directory not found!
    echo Please run setup_portable.bat first
    echo.
    pause
    exit /b 1
)

REM Check if Python is properly configured
echo Verifying Python configuration...
portable_python\python.exe -c "import sys; print('Python path:', sys.path)" 2>nul
if errorlevel 1 (
    echo Python configuration issue detected. Fixing...
    echo.
    
    REM Recreate the .pth file with correct configuration
    REM The embeddable package stores stdlib in python311.zip
    (
        echo python311.zip
        echo .
        echo Lib
        echo Lib\site-packages
        echo import site
    ) > portable_python\python311._pth
    
    echo Python configuration fixed.
    echo.
    echo Please run install_deps.bat again to apply the fix.
    pause
    exit /b 1
)

REM Test Python again
portable_python\python.exe --version 2>nul
if errorlevel 1 (
    echo ERROR: Python still not working correctly
    echo Please delete portable_python folder and run setup_portable.bat again
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
