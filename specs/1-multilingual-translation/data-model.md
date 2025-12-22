# Data Model: Multilingual Translation

## Entities

### LanguagePreference
**Description**: User's selected language preference stored in browser

**Attributes**:
- `selectedLanguage` (string): The ISO language code (e.g., 'en', 'ur', 'fr', 'de', 'es', 'zh', 'hi', 'ja')
- `timestamp` (datetime): When the preference was last updated
- `fallbackLanguage` (string): Default language ('en') when translations fail

### TranslationCache
**Description**: In-memory cache for storing recent translations

**Attributes**:
- `sourceText` (string): Original text to be translated
- `sourceLanguage` (string): Language code of source text ('auto' for detection)
- `targetLanguage` (string): Language code of translation
- `translatedText` (string): Translated text result
- `timestamp` (datetime): When the translation was cached
- `ttl` (integer): Time-to-live in seconds for cache invalidation

### TranslationRequest
**Description**: Structure for API requests to translation service

**Attributes**:
- `q` (string): Text to translate (up to 5000 characters)
- `source` (string): Source language code (e.g., 'en')
- `target` (string): Target language code (e.g., 'ur')
- `format` (string): Text format ('text' or 'html')
- `api_key` (string): API key if required (not needed for LibreTranslate)

### TranslationResponse
**Description**: Structure for responses from translation service

**Attributes**:
- `translatedText` (string): The translated text
- `detectedSourceLanguage` (string): Language detected from source text
- `status` (string): Success or error status
- `errorMessage` (string): Error details if translation failed