@echo off
echo ========================================
echo Prompt Enhancer - Environment Check
echo ========================================
echo.

REM Check portable Python
echo [1/5] Checking portable Python...
if exist "portable_python\python.exe" (
    echo ✓ portable_python/python.exe found
    portable_python\python.exe --version
    
    REM Get detailed Python version info
    echo Version details:
    portable_python\python.exe -c "import sys; print('  Version:', sys.version_info[:2])"
    portable_python\python.exe -c "import sys; print('  Full version:', sys.version.split()[0])"
    portable_python\python.exe -c "import sys; print('  Python 3.14 compatible:' if sys.version_info >= (3, 14) else '  Consider upgrading to Python 3.14')"
) else (
    echo ✗ portable_python/python.exe NOT found
    echo Please run setup_portable.bat first
    echo.
    pause
    exit /b 1
)
echo.

REM Check dependencies
echo [2/5] Checking dependencies...
set NEED_DEPS=0

portable_python\python.exe -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ✗ FastAPI not installed
    set NEED_DEPS=1
) else (
    echo ✓ FastAPI installed
)

portable_python\python.exe -c "
try:
    import google.genai
    print('✓ google.genai (Python 3.14 compatible) installed')
except ImportError:
    try:
        import google.generativeai
        print('✓ google.generativeai (legacy) installed')
    except ImportError:
        print('✗ Google GenAI not installed')
        exit(1)
" 2>nul
if errorlevel 1 set NEED_DEPS=1

portable_python\python.exe -c "import librosa" 2>nul
if errorlevel 1 (
    echo ✗ Librosa not installed
    set NEED_DEPS=1
) else (
    echo ✓ Librosa installed
)

portable_python\python.exe -c "import numpy" 2>nul
if errorlevel 1 (
    echo ✗ NumPy not installed
    set NEED_DEPS=1
) else (
    echo ✓ NumPy installed
)

portable_python\python.exe -c "import scipy" 2>nul
if errorlevel 1 (
    echo ✗ SciPy not installed
    set NEED_DEPS=1
) else (
    echo ✓ SciPy installed
)
echo.

REM Check API key
echo [3/5] Checking API key...
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

REM Check Python 3.14 specific compatibility
echo [4/5] Checking Python 3.14 compatibility...
portable_python\python.exe -c "
import sys
py_version = sys.version_info[:2]
if py_version >= (3, 14):
    print('✓ Python 3.14+ - Latest features available')
    print('  ✓ Enhanced Google GenAI compatibility')
    print('  ✓ Latest Python optimizations')
elif py_version == (3, 11):
    print('✓ Python 3.11 - Stable configuration')
    print('  ✓ Full compatibility maintained')
    print('  ℹ Consider upgrading to Python 3.14')
else:
    print('⚠ Unsupported Python version:', py_version)
    print('  Please use Python 3.11 or 3.14')
"

echo.

REM Check directories
echo [5/5] Checking directories...
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

if exist "app\static\data\themes.json" (
    echo ✓ Theme data file found
) else (
    echo ✗ Theme data file NOT found
)

if exist "app\static\data\templates.json" (
    echo ✓ Template data file found
) else (
    echo ✗ Template data file NOT found
)
echo.

REM Summary
echo ========================================
echo ENVIRONMENT SUMMARY:
echo ========================================
echo Python Version:
portable_python\python.exe -c "import sys; print('  ' + sys.version.split()[0] + ' (' + '.'.join(map(str, sys.version_info[:2])) + ')')"

echo.
echo Dependencies Status:
if defined NEED_DEPS (
    echo ✗ Some dependencies missing
    echo   Action: Run install_deps.bat
) else (
    echo ✓ All dependencies installed
)

echo.
echo Compatibility:
portable_python\python.exe -c "
import sys
if sys.version_info >= (3, 14):
    print('✓ Python 3.14 compatible - Latest features')
else:
    print('✓ Python 3.11 compatible - Stable')
"

echo.
echo Recommendations:
portable_python\python.exe -c "
import sys
if sys.version_info < (3, 14):
    print('ℹ Consider upgrading to Python 3.14 for:')
    print('  - Latest Python features')
    print('  - Enhanced performance')
    print('  - Better Google GenAI compatibility')
else:
    print('✓ Using latest Python 3.14!')
    print('✓ All features available')
"

echo.
if defined NEED_DEPS (
    echo Action required: Run install_deps.bat to install missing dependencies
) else (
    echo ✓ Everything looks good! You can run run_app.bat
)
echo.

pause
