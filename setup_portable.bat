@echo off
echo ========================================
echo Prompt Enhancer - Python Setup
echo ========================================
echo.
echo Choose Python version:
echo 1. Python 3.11.7 (Stable, tested)
echo 2. Python 3.14.3 (Latest, recommended)
echo.
set /p python_choice="Enter choice (1 or 2): "

if "%python_choice%"=="1" goto setup_python311
if "%python_choice%"=="2" goto setup_python314
echo Invalid choice. Please select 1 or 2.
echo.
pause
exit /b 1

:setup_python311
echo Setting up Python 3.11.7...
set PYTHON_VERSION=3.11.7
set PYTHON_EMBED=python-3.11.7-embed-amd64.zip
set PYTHON_PTH=python311._pth
set PYTHON_ZIP=python311.zip
goto download_python

:setup_python314
echo Setting up Python 3.14.3...
set PYTHON_VERSION=3.14.3
set PYTHON_EMBED=python-3.14.3-embed-amd64.zip
set PYTHON_PTH=python314._pth
set PYTHON_ZIP=python314.zip
goto download_python

:download_python
REM Check if portable Python already exists
if exist "portable_python\python.exe" (
    echo Portable Python already found!
    echo You can run install_deps.bat and then run_app.bat
    echo.
    pause
    exit /b 0
)

echo Downloading Python %PYTHON_VERSION% embeddable package automatically...
echo This may take a few minutes depending on your internet speed.
echo.

REM Create portable_python directory
if not exist "portable_python" mkdir portable_python

REM Download Python embeddable package
echo [1/3] Downloading Python %PYTHON_VERSION% embeddable package...
powershell -Command "& {$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_EMBED%' -OutFile 'python-embed.zip'}"

if not exist "python-embed.zip" (
    echo ERROR: Failed to download Python package
    echo Please check your internet connection and try again
    echo.
    pause
    exit /b 1
)

echo Download complete!
echo.

REM Extract the ZIP file
echo [2/3] Extracting Python package...
powershell -Command "Expand-Archive -Path 'python-embed.zip' -DestinationPath 'portable_python' -Force"

if not exist "portable_python\python.exe" (
    echo ERROR: Failed to extract Python package
    echo Please try running the script again
    echo.
    pause
    exit /b 1
)

echo Extraction complete!
echo.

REM Clean up the downloaded ZIP
echo [3/3] Cleaning up...
del python-embed.zip

REM Configure Python for pip (fix embeddable package)
echo Configuring Python for package installation...

REM The embeddable package stores stdlib in pythonXXX.zip
REM We need: pythonXXX.zip, current dir, Lib, and import site
(
    echo %PYTHON_ZIP%
    echo .
    echo Lib
    echo Lib\site-packages
    echo import site
) > portable_python\%PYTHON_PTH%

REM Verify Python works
echo Testing Python configuration...
portable_python\python.exe --version
if errorlevel 1 (
    echo ERROR: Python configuration failed
    echo Please delete portable_python folder and try again
    pause
    exit /b 1
)

echo Python configuration successful!

REM Install pip
echo Installing pip...
portable_python\python.exe -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')"
portable_python\python.exe get-pip.py --no-warn-script-location
del get-pip.py

REM Check for .env file
if not exist ".env" (
    echo.
    echo ========================================
    echo API KEY SETUP REQUIRED
    echo ========================================
    echo.
    echo You need a Google API key to use this application.
    echo.
    echo 1. Go to: https://aistudio.google.com/app/apikey
    echo 2. Create a new API key
    echo 3. Create a .env file with: GOOGLE_API_KEY=your_key_here
    echo.
    echo Would you like to create a .env file template now? (Y/N)
    set /p create_env=
    if /i "%create_env%"=="Y" (
        echo GOOGLE_API_KEY=your_api_key_here > .env
        echo Created .env file template.
        echo Please edit it with your actual API key.
    )
)

echo.
echo ========================================
echo SUCCESS: Portable Python %PYTHON_VERSION% setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your Google API key (if not done)
echo 2. Run install_deps.bat
echo 3. Run run_app.bat
echo.
pause
