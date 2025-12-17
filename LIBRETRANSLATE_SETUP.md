# LibreTranslate Setup Guide

This guide will help you set up LibreTranslate for your humanoid chatbot book application.

## Important: Self-Hosting is Required for Free Usage

**Public instances often have rate limits or restrictions. To have truly unlimited free translation, you must self-host LibreTranslate.**

## Self-Hosted LibreTranslate (Required for Free Usage)

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps (Required)

1. Install LibreTranslate:
```bash
pip install libretranslate
```

2. Start the service:
```bash
libretranslate --host 0.0.0.0 --port 5000
```

3. Your LibreTranslate instance will be available at `http://localhost:5000`

4. Update your `.env` file:
```
REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate
```

### Alternative Installation Methods

#### Using Docker (Recommended):
```bash
docker run -d -p 5000:5000 --name libretranslate libretranslate/libretranslate
```

#### Using Docker Compose:
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  libretranslate:
    image: libretranslate/libretranslate
    ports:
      - "5000:5000"
    volumes:
      - ./data:/data
    restart: unless-stopped
```

Then run:
```bash
docker-compose up -d
```

## Option 2: Alternative Translation Service (Recommended for Windows)

If Docker is not available on your system (Windows without Docker Desktop), you can use the deep-translator library as an alternative:

1. Install the deep-translator library:
```bash
pip install deep-translator
```

2. Update your `.env` file to use deep-translator:
```
TRANSLATION_SERVICE=deep_translator
REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate
```

3. Update the backend translation service to use deep-translator (already configured in this project)

4. The application will use the deep-translator library instead of LibreTranslate for translation services.

## Option 3: Public LibreTranslate Instance (Limited/Not Recommended)

⚠️ **Note**: Public instances may have rate limits, restrictions, or may become unavailable.

1. Copy the `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Your `.env` file should have:
```
REACT_APP_LIBRETRANSLATE_URL=https://translate.terraprint.co/translate
TRANSLATION_SERVICE=libretranslate
```

3. The application will use the public instance, but may encounter limitations.

## Configuration for Your Application

1. Make sure your `.env` file contains:
```
REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate
```
(adjust the URL based on where your LibreTranslate instance is running)

2. Restart your frontend application:
```bash
cd frontend
npm start
```

## Verification

To verify LibreTranslate is working:

1. Visit `http://localhost:5000` in your browser (if self-hosted)
2. You should see the LibreTranslate interface
3. Test with a simple request:
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "q": "Hello, world!",
    "source": "en",
    "target": "es"
  }'
```

## Troubleshooting

### Common Issues:

1. **Connection Refused**: Make sure LibreTranslate is running and accessible
2. **CORS Issues**: LibreTranslate should handle CORS automatically
3. **Rate Limits**: Self-hosted instances don't have rate limits

### Environment Variables:

- `REACT_APP_LIBRETRANSLATE_URL`: URL to your LibreTranslate instance
- Make sure to restart your React app after changing environment variables

## Supported Languages

LibreTranslate supports all the languages in your application:
- English (en)
- Urdu (ur)
- Hindi (hi)
- Spanish (es)
- Japanese (ja)
- Chinese (zh)
- French (fr)
- German (de)

## Production Considerations

For production use:
- Use self-hosted instance for reliability
- Set up HTTPS with a reverse proxy (nginx)
- Configure proper resource limits
- Set up monitoring and logging
- Consider using a subdomain (e.g., translate.yourdomain.com)

## Alternative Public Instances

If you prefer not to self-host, you can try these public instances:
- `https://libretranslate.com/translate` (official)
- `https://translate.terraprint.co/translate` (community)

⚠️ **Note**: Public instances may have usage limits and are not guaranteed to be available.