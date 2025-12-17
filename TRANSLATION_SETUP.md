# Translation Setup Guide

This guide explains how to set up translation functionality for the humanoid chatbot book application.

## Supported Languages

The application currently supports translation for the following languages:
- English (en) 🇬🇧
- Urdu (ur) 🇵🇰
- Hindi (hi) 🇮🇳
- Spanish (es) 🇪🇸
- Japanese (ja) 🇯🇵
- Chinese (zh) 🇨🇳
- French (fr) 🇫🇷
- German (de) 🇩🇪

## Translation Services

The application supports multiple translation services with fallback options:

### 1. Google Translate API (Recommended)
- Most accurate translations
- Supports all languages
- Requires API key

To set up:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Translate API
4. Create an API key
5. Add the key to your `.env` file:

```
REACT_APP_GOOGLE_TRANSLATE_API_KEY=your_api_key_here
```

### 2. Azure Translator (Alternative)
- Good translation quality
- Supports most languages
- Requires API key and region

To set up:
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create a Translator resource
3. Get the key and region
4. Add to your `.env` file:

```
REACT_APP_AZURE_TRANSLATOR_KEY=your_azure_key_here
REACT_APP_AZURE_TRANSLATOR_REGION=your_region_here
```

### 3. LibreTranslate (Free/Open Source)
- Free to use public instances
- Self-hostable
- Good for development/testing

To use public instance:
```
REACT_APP_LIBRETRANSLATE_URL=https://libretranslate.com/translate
```

To use self-hosted instance:
```
REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Add your API keys to the `.env` file

3. Restart your development server:
```bash
cd frontend
npm start
```

## Components

### TranslationButton
- Dropdown menu for switching between languages
- Supports all 8 languages with flags
- Changes URL structure to match selected language

### ChapterTranslation
- Translates book content in real-time
- Shows translated content in a separate section
- Supports all languages with error handling

### TranslationPanel
- Full-featured translation interface
- Supports source and target language selection
- Translation history
- Copy to clipboard functionality

## Docusaurus Integration

The application uses Docusaurus' built-in i18n support. When a user selects a language:

1. The URL changes to include the language code (e.g., `/es/`, `/ur/`, `/hi/`)
2. Docusaurus serves the appropriate localized content if available
3. Dynamic content is translated using the translation service

## Troubleshooting

### Common Issues:

1. **API Key Not Working**:
   - Verify the API key is correctly set in `.env`
   - Check that the API service is enabled in your cloud console
   - Ensure your IP address is whitelisted if required

2. **Translation Not Working**:
   - Check browser console for error messages
   - Verify network connectivity
   - Ensure the `.env` file is properly loaded (restart the development server)

3. **Missing Translations**:
   - Some languages might have incomplete static content
   - Dynamic content will always be translated via the API
   - Consider contributing translations to the static content

## Development Notes

- The translation service includes fallback mechanisms
- If all APIs fail, a mock translation is returned
- Translation requests are rate-limited to respect API quotas
- Error handling is implemented across all translation components

## Security

- API keys should never be committed to version control
- Use environment variables to store sensitive information
- Consider using backend proxy for production to hide API keys