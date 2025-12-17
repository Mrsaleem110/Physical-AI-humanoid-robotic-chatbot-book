# Quick Setup for Translation Services (Truly Free Setup)

## Step 1: Choose Your Translation Method

You have two options for translation services:

### Option A: Self-Host LibreTranslate (For Docker users)

**Important: Public instances often have rate limits. Self-hosting is required for unlimited free usage.**

#### Install LibreTranslate:
```bash
# Install using pip
pip install libretranslate

# Start the service (required for free usage)
libretranslate --host 0.0.0.0 --port 5000
```

Or using Docker:
```bash
docker run -d -p 5000:5000 --name libretranslate libretranslate/libretranslate
```

### Option B: Use deep-translator Library (Recommended for Windows without Docker)

For Windows systems without Docker, use the deep-translator library as an alternative:

#### Install deep-translator:
```bash
# Install using pip
pip install deep-translator

# Update backend requirements
cd backend
pip install -r requirements.txt
```

## Step 2: Configure Your Application

For both options, configure your environment:

```bash
# Copy environment file
cp .env.example .env

# For Option A (LibreTranslate), the .env is already configured:
# REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate

# For Option B (deep-translator), ensure this setting in .env:
TRANSLATION_SERVICE=deep_translator

# Start your frontend
cd frontend
npm install
npm start
```

## Alternative: Public Instance (Limited)

⚠️ **Note**: This may have rate limits or restrictions:

```bash
# Copy environment file
cp .env.example .env

# Edit .env and uncomment a public instance:
# REACT_APP_LIBRETRANSLATE_URL=https://translate.terraprint.co/translate

# Start your frontend
cd frontend
npm install
npm start
```

## Verify Installation:
1. Open your browser to http://localhost:3000 (your app)
2. Click the translation button
3. Select any language
4. The content should translate in real-time

## Troubleshooting:
- If using LibreTranslate: Make sure your instance allows requests from localhost:3000
- If using deep-translator: Ensure the backend translation service is properly configured
- If translations fail, check the browser console and backend logs for error messages
- Make sure to restart your frontend after changing environment variables