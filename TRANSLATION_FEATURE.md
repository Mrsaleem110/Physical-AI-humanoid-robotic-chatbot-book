# English-Urdu Translation Feature

This feature provides comprehensive bilingual support for the website, allowing users to switch between English and Urdu with a single click.

## Supported Languages

- 🇬🇧 English (en)
- 🇵🇰 Urdu (ur) - with RTL support

## How It Works

1. **Dynamic Translation**: The system dynamically translates all text content on the page in real-time
2. **Smart Caching**: Translations are cached to improve performance on subsequent visits
3. **RTL Support**: Right-to-left layout for Urdu is properly supported
4. **SEO Friendly**: Maintains proper language attributes for search engines
5. **Persistent Preferences**: Remembers user's language preference using localStorage and URL parameters

## Implementation

The translation system is implemented using:

- **Translation Context**: Manages global translation state
- **Translation Service**: Handles API calls to translation services (Google Translate, LibreTranslate, etc.)
- **Docusaurus Integration**: Works seamlessly with Docusaurus i18n features
- **Custom Components**: Provides translation UI elements throughout the site

## Usage

The translation functionality is automatically available throughout the website:

1. **Navigation**: Use the language dropdown in the top navigation bar
2. **Automatic Detection**: The system remembers your preference and applies it automatically
3. **Page-level Translation**: All content on the current page is translated instantly

## Technical Details

- **Context Provider**: Wrapped around the entire application in `src/theme/wrapper.js`
- **Service Integration**: Uses multiple translation APIs with fallbacks
- **Performance**: Implements smart caching and batch translation for efficiency
- **Accessibility**: Maintains proper language attributes and directionality

## API Integration

The system supports multiple translation services:
- Google Translate API (requires API key)
- Azure Translator (requires API key)
- LibreTranslate (free, open-source)
- Mock translation for development

## Configuration

Environment variables for translation services:
- `REACT_APP_LIBRETRANSLATE_URL` - LibreTranslate API URL
- `REACT_APP_GOOGLE_TRANSLATE_API_KEY` - Google Translate API key
- `REACT_APP_AZURE_TRANSLATOR_KEY` - Azure Translator API key
- `REACT_APP_AZURE_TRANSLATOR_REGION` - Azure region