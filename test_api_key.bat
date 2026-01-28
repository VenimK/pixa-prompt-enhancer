@echo off
echo ========================================
echo Testing Google API Key
echo ========================================
echo.

REM Check if .env file exists
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please create .env file with your Google API key
    echo.
    pause
    exit /b 1
)

REM Check if portable Python exists
if not exist "portable_python\python.exe" (
    echo ERROR: portable Python not found!
    echo Please run setup_portable.bat first
    echo.
    pause
    exit /b 1
)

echo Testing your Google API key...
echo.

REM Create test script
(
echo import os
echo from dotenv import load_dotenv
echo import google.genai as genai
echo.
echo load_dotenv^(^)
echo.
echo api_key = os.environ.get^("GOOGLE_API_KEY"^)
echo if not api_key:
echo     print^("ERROR: GOOGLE_API_KEY not found in .env file"^)
echo     exit^(1^)
echo.
echo print^(f"API Key found: {api_key[:10]}...{api_key[-4:]}"^)
echo print^(f"API Key length: {len(api_key)} characters"^)
echo.
echo try:
echo     client = genai.Client^(api_key=api_key^)
echo     response = client.models.generate_content^(model="gemini-2.5-flash", contents="Hello"^)
echo     print^("SUCCESS: API key is working correctly!"^)
echo     print^(f"Response: {response.text[:100]}..."^)
echo except Exception as e:
echo     print^(f"ERROR: {e}"^)
echo     print^("")
echo     print^("Common solutions:"^)
echo     print^("1. Check if API key is enabled for Gemini API"^)
echo     print^("2. Verify no extra spaces or quotes in .env file"^)
echo     print^("3. Create new API key at https://aistudio.google.com/app/apikey"^)
) > test_api.py

REM Run the test
portable_python\python.exe test_api.py

REM Clean up
del test_api.py

echo.
pause
