Prompt Enhancer - Windows Setup Instructions
=============================================

This version includes automatic Python setup - no installation required!

QUICK START (One Click Setup):
------------------------------
1. Double-click: run_app.bat
   - Automatically downloads Python (first time only)
   - Installs all dependencies 
   - Starts the application
   - Browser opens at http://localhost:8000

That's it! Everything is handled automatically.

DETAILED STEPS:
--------------
If you prefer manual control:

1. Double-click: setup_portable.bat
   - Downloads Python embeddable package automatically
   - Extracts and configures portable environment

2. Double-click: install_deps.bat  
   - Installs required packages (FastAPI, Google AI, etc.)

3. Double-click: run_app.bat
   - Starts the application
   - Browser opens automatically

WHAT HAPPENS AUTOMATICALLY:
---------------------------
- Downloads Python 3.11.7 embeddable package (~25MB)
- Extracts to portable_python/ folder
- Configures Python for package installation
- Installs all dependencies from requirements.txt
- Creates necessary directories
- Launches FastAPI web server

TROUBLESHOOTING:
---------------
- If download fails: Check internet connection and run again
- If setup fails: Delete portable_python folder and try again
- If app won't start: Run check_environment.bat for diagnosis
- If dependencies fail: Run install_deps.bat manually

REQUIREMENTS:
------------
- Windows 10/11 (64-bit)
- Internet connection (for first-time setup)
- No Python installation needed

INCLUDED FEATURES:
-----------------
- Automatic Python download and setup
- FastAPI web server
- Image analysis and enhancement
- Audio analysis capabilities  
- Google Gemini AI integration
- Self-contained deployment
- One-click startup

FOLDER STRUCTURE:
----------------
portable_python/     # Auto-downloaded Python (created during setup)
app/                 # Application code
uploads/             # Temporary file uploads
requirements.txt     # Python dependencies
.env                # API keys and configuration
setup_portable.bat   # Automatic Python setup
install_deps.bat     # Install Python dependencies
run_app.bat          # Start the application (one-click)
check_environment.bat # Diagnose issues

PERFORMANCE NOTES:
-----------------
- First run: 2-5 minutes (downloads and setup)
- Subsequent runs: 10-20 seconds (startup only)
- No impact on system Python installation
- Completely isolated environment

For support or updates, check the project repository.
