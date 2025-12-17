#!/bin/bash
# LibreTranslate Setup Script for Humanoid Chatbot Book

echo "Setting up LibreTranslate for the Humanoid Chatbot Book..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is not installed. Please install pip."
    exit 1
fi

echo "Installing LibreTranslate..."
pip install libretranslate

echo "Starting LibreTranslate service..."
libretranslate --host 0.0.0.0 --port 5000 &

LIBRE_PID=$!
echo "LibreTranslate started with PID: $LIBRE_PID"

echo "LibreTranslate is now running on http://localhost:5000"
echo "You can now start your frontend application in another terminal:"

cat << 'EOF'
cd frontend
npm install
npm start
EOF

echo ""
echo "To stop LibreTranslate, run: kill $LIBRE_PID"
echo ""
echo "For Docker setup (recommended), run:"
echo "docker run -d -p 5000:5000 --name libretranslate libretranslate/libretranslate"