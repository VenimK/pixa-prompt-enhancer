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

REM Detect Python version
echo Detecting Python version...
portable_python\python.exe -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" > python_version.txt
set /p PYTHON_VERSION=<python_version.txt
del python_version.txt

echo Detected Python %PYTHON_VERSION%
echo.

REM Check if Python is properly configured
echo Verifying Python configuration...
portable_python\python.exe -c "import sys; print('Python path:', sys.path)" 2>nul
if errorlevel 1 (
    echo Python configuration issue detected. Fixing...
    echo.
    
    REM Recreate the .pth file with correct configuration
    REM The embeddable package stores stdlib in pythonXXX.zip
    if "%PYTHON_VERSION%"=="3.11" (
        echo Creating python311._pth for Python 3.11...
        (
            echo python311.zip
            echo .
            echo Lib
            echo Lib\site-packages
            echo import site
        ) > portable_python\python311._pth
    ) else if "%PYTHON_VERSION%"=="3.14" (
        echo Creating python314._pth for Python 3.14...
        (
            echo python314.zip
            echo .
            echo Lib
            echo Lib\site-packages
            echo import site
        ) > portable_python\python314._pth
    ) else (
        echo Unsupported Python version: %PYTHON_VERSION%
        echo Please use Python 3.11 or 3.14
        pause
        exit /b 1
    )
    
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

echo Found portable Python %PYTHON_VERSION%, installing dependencies...
echo.

REM Check Python 3.14 compatibility
if "%PYTHON_VERSION%"=="3.14" (
    echo ✓ Python 3.14 detected - Using latest Google GenAI package
    echo ✓ Enhanced performance and features available
    echo.
) else if "%PYTHON_VERSION%"=="3.11" (
    echo ✓ Python 3.11 detected - Using stable configuration
    echo ✓ Consider upgrading to Python 3.14 for latest features
    echo.
)

REM Upgrade pip first
portable_python\python.exe -m pip install --upgrade pip

REM Install dependencies
portable_python\python.exe -m pip install -r requirements.txt

REM Verify Google GenAI installation
echo.
echo Verifying Google GenAI installation...
portable_python\python.exe -c "
try:
    import google.genai
    print('✓ google.genai (Python 3.14 compatible) installed')
except ImportError:
    try:
        import google.generativeai
        print('✓ google.generativeai (legacy) installed')
    except ImportError:
        print('✗ Google GenAI package not found')
"

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Python %PYTHON_VERSION% with all dependencies installed!
echo.
if "%PYTHON_VERSION%"=="3.14" (
    echo ✓ Latest Python 3.14 features available
    echo ✓ Enhanced Google GenAI compatibility
) else (
    echo ✓ Stable Python 3.11 configuration
    echo ✓ Full compatibility maintained
)
echo.
echo You can now run the app with run_app.bat
echo.
pause
