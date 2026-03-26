Prompt Enhancer - Windows Setup Instructions
=============================================

This version includes automatic Python setup - no installation required!

QUICK START (One Click Setup):
------------------------------
1. Double-click: run_app.bat
   - Automatically downloads Python (first time only)
   - Installs all dependencies 
   - Starts the application
   - Browser opens at http://localhost:8002

That's it! Everything is handled automatically.

API KEY SETUP (Required First Time):
-----------------------------------
The app needs a Google API key to work:

1. **Get API Key:**
   - Go to: https://aistudio.google.com/app/apikey
   - Click "Create API Key"
   - Copy the generated key

2. **Create .env File:**
   - Open Notepad
   - Type: GOOGLE_API_KEY=your_actual_api_key_here
   - Save as: .env (in the app folder)
   - **Important:** Choose "All Files (*.*)" as save type

3. **Alternative Methods:**
   - Command Prompt: echo GOOGLE_API_KEY=your_key_here > .env
   - PowerShell: "GOOGLE_API_KEY=your_key_here" | Out-File .env

QUICK START (Windows):
----------------------
1. Double-click: setup_portable.bat  
   - Choose Python 3.11.7 (stable) or Python 3.14.3 (latest)
   - Downloads and configures Python automatically

2. Double-click: install_deps.bat  
   - Installs required packages (FastAPI, Google AI, etc.)

3. Double-click: run_app.bat
   - Starts the application
   - Browser opens automatically

WHAT HAPPENS AUTOMATICALLY:
---------------------------
- Downloads Python 3.11+ (Python 3.14 supported) embeddable package (~25MB)
- Extracts to portable_python/ folder
- Configures Python for package installation
- Installs pip package manager
- Installs all dependencies from requirements.txt
- Creates necessary directories
- Launches FastAPI web server

PYTHON VERSION OPTIONS:
-----------------------
• Python 3.11.7 - Stable, thoroughly tested
• Python 3.14.3 - Latest, recommended for new features
  ✓ Enhanced Google GenAI compatibility
  ✓ Latest Python optimizations
  ✓ Better performance

FEATURES:
---------
• AI-Powered Prompt Enhancement
• 8 Professional Themes
• Advanced Theme Editor
• Export Themes (JSON, CSS, SCSS)
• Python 3.14 Support
• Google GenAI Integration
• Image & Audio Processing

TROUBLESHOOTING:
---------------
- If download fails: Check internet connection and run again
- If Python errors: Delete portable_python folder and retry
- If API errors: Check .env file contains valid GOOGLE_API_KEY
- For Python 3.14: Uses google-genai package (latest)
- For Python 3.11: Uses google-generativeai package (legacy)
- If setup fails: Delete portable_python folder and try again
- If app won't start: Run check_environment.bat for diagnosis
- If dependencies fail: Run install_deps.bat manually
- If API key error: 
  * Run test_api_key.bat to diagnose the issue
  * Check .env file format (no quotes, no spaces)
  * Enable Gemini API at https://aistudio.google.com/app/apikey
  * Create new API key if needed

REQUIREMENTS:
------------
- Windows 10/11 (64-bit)
- Internet connection (for first-time setup)
- Google API key (free from https://aistudio.google.com/app/apikey)
- No Python installation needed

INCLUDED FEATURES:
-----------------
- Automatic Python download and setup
- Automatic .env file creation assistance
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
.env                # API keys and configuration (you create this)
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
