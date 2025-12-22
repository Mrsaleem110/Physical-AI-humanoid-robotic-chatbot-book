# Multilingual Translation Feature

## Overview
The multilingual translation feature allows users to dynamically translate website content into multiple languages using the LibreTranslate API. The feature provides a language selector dropdown, real-time translation of both static and dynamic content, and persistence of language preferences across sessions.

## Supported Languages
- English (en)
- Urdu (ur)
- French (fr)
- German (de)
- Spanish (es)
- Chinese (zh)
- Hindi (hi)
- Japanese (ja)

## Components

### LanguageSelector
A dropdown component that allows users to select their preferred language. Displays flags and language names for better user experience.

### TranslationProvider
A React context provider that manages the translation state, language preferences, and translation functions across the application.

### TranslateText
A component for translating individual text elements. Automatically translates content based on the selected language.

### TranslateContent
A component for translating larger content blocks like blog posts or book chapters, preserving HTML structure while translating text content.

## Usage

### Basic Setup
Wrap your application with the TranslationProvider:

```jsx
import { TranslationProvider } from './components/Translation/TranslationProvider';

function App() {
  return (
    <TranslationProvider defaultLanguage="en">
      {/* Your application components */}
      <LanguageSelector />
    </TranslationProvider>
  );
}
```

### Translating Text
Use the TranslateText component for individual text elements:

```jsx
import TranslateText from './components/Translation/TranslateText';

function MyComponent() {
  return (
    <div>
      <h1><TranslateText>Welcome to our site</TranslateText></h1>
      <p><TranslateText>This content will be translated</TranslateText></p>
    </div>
  );
}
```

### Using the Hook
Use the useTranslation hook for programmatic access to translation functions:

```jsx
import { useTranslation } from './hooks/useTranslation';

function MyComponent() {
  const { translate, currentLanguage, changeLanguage } = useTranslation();

  return (
    <div>
      <p>Current language: {currentLanguage}</p>
      <button onClick={() => changeLanguage('es')}>
        Switch to Spanish
      </button>
    </div>
  );
}
```

## Configuration
Set the LibreTranslate API URL in your environment variables:
```
REACT_APP_LIBRETRANSLATE_URL=https://libretranslate.com
REACT_APP_LIBRETRANSLATE_API_KEY=your_api_key_if_required
```

## Features
- Real-time translation without page reloads
- Language preference persistence using localStorage
- Caching to reduce API calls and improve performance
- Fallback to original text if translation fails
- Support for both static and dynamic content
- Responsive UI with loading indicators
- Error handling and graceful degradation

## Performance Considerations
- Translation caching to avoid repeated API calls
- Batch processing for multiple text elements
- Loading states to provide user feedback
- Efficient HTML parsing for content translation