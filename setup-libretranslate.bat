@echo off
echo Setting up LibreTranslate for the Humanoid Chatbot Book...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo Installing LibreTranslate...
pip install libretranslate

echo Starting LibreTranslate service...
start cmd /k "libretranslate --host 0.0.0.0 --port 5000"

echo LibreTranslate is now running on http://localhost:5000
echo You can now start your frontend application in another terminal:

echo cd frontend
echo npm install
echo npm start

echo.
echo For Docker setup (recommended), run:
echo docker run -d -p 5000:5000 --name libretranslate libretranslate/libretranslate

pause