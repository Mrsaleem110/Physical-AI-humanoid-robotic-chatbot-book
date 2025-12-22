# Quickstart: Multilingual Translation

## Setup

1. Initialize the translation provider in your application:
   ```javascript
   import { TranslationProvider } from './src/components/Translation/TranslationProvider';

   // Wrap your app with the provider
   <TranslationProvider defaultLanguage="en">
     <App />
   </TranslationProvider>
   ```

## Basic Usage

### Adding Language Selector
```javascript
import LanguageSelector from './src/components/Translation/LanguageSelector';

// Add the language selector to your UI
<LanguageSelector />
```

### Translating Content
```javascript
import { useTranslation } from './src/hooks/useTranslation';

function MyComponent() {
  const { translate, currentLanguage } = useTranslation();

  return (
    <div>
      <h1>{translate('Welcome to our site')}</h1>
      <p>{translate('This content will be translated')}</p>
    </div>
  );
}

// Or use the TranslateText component for simple text translation:
import TranslateText from './src/components/Translation/TranslateText';

function MyComponent() {
  return (
    <div>
      <h1><TranslateText>Welcome to our site</TranslateText></h1>
      <p><TranslateText>This content will be translated</TranslateText></p>
    </div>
  );
}
```

## Configuration

Set up your LibreTranslate API endpoint in your environment variables:
```
REACT_APP_LIBRETRANSLATE_URL=https://libretranslate.com
REACT_APP_LIBRETRANSLATE_API_KEY=your_api_key_if_required
```

## Supported Languages
- English (en)
- Urdu (ur)
- French (fr)
- German (de)
- Spanish (es)
- Chinese (zh)
- Hindi (hi)
- Japanese (ja)

## Testing
Run the following to verify the translation functionality:
```bash
npm run test:translation
```

## Troubleshooting
- If translations aren't appearing, check that the LibreTranslate service is running
- For performance issues, verify that translation caching is working properly
- If language persistence isn't working, check browser localStorage permissions
- Make sure the TranslationProvider wraps all components that need translation