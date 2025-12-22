// Translation Service for Multilingual Translation Feature
// This service handles translation of page content using LibreTranslate API

class TranslationService {
  constructor() {
    // LibreTranslate API configuration
    this.libreTranslateUrl = process.env.REACT_APP_LIBRETRANSLATE_URL || 'https://libretranslate.de';
    this.apiKey = process.env.REACT_APP_LIBRETRANSLATE_API_KEY || null;
  }

  // Function to translate text using multiple free translation APIs
  async translateText(text, targetLang, sourceLang = 'auto') {
    if (!text || text.trim().length === 0) {
      return text;
    }

    // For Korean, we'll use a different approach to avoid quota issues
    // Since we have proper i18n files for Korean, prioritize Docusaurus i18n over API calls
    if (targetLang === 'ko') {
      // Return original text for Korean since we have proper i18n support
      // The actual Korean content will be served via Docusaurus i18n system
      return text;
    }

    // Try MyMemory API first (free, no key required)
    try {
      const myMemoryUrl = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${sourceLang}|${targetLang}`;
      const myMemoryResponse = await fetch(myMemoryUrl);

      if (myMemoryResponse.ok) {
        const myMemoryData = await myMemoryResponse.json();
        if (myMemoryData && myMemoryData.responseData && myMemoryData.responseData.translatedText) {
          return myMemoryData.responseData.translatedText;
        }
      }
    } catch (error) {
      console.error('MyMemory API error:', error);
    }

    // Fallback to LibreTranslate API
    try {
      const response = await fetch(`${this.libreTranslateUrl}/translate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          q: text,
          source: sourceLang,
          target: targetLang,
          format: 'text',
          api_key: this.apiKey || undefined
        })
      });

      if (!response.ok) {
        throw new Error(`LibreTranslate API error: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();

      // Handle different response formats from LibreTranslate instances
      if (data.translatedText !== undefined) {
        return data.translatedText;
      } else if (data[0] && data[0].translatedText !== undefined) {
        return data[0].translatedText;
      } else {
        throw new Error('Unexpected response format from LibreTranslate service');
      }
    } catch (error) {
      console.error('Translation error:', error);
      // Return original text if translation fails - fallback to English requirement
      return text;
    }
  }

  // Function to get supported languages from LibreTranslate API
  async getSupportedLanguages() {
    try {
      const response = await fetch(`${this.libreTranslateUrl}/languages`);

      if (!response.ok) {
        throw new Error(`Languages API error: ${response.status} - ${response.statusText}`);
      }

      const languages = await response.json();

      // Map to our expected format
      return languages.map(lang => ({
        code: lang.code,
        name: lang.name
      }));
    } catch (error) {
      console.error('Error fetching supported languages:', error);
      // Return default languages if API fails
      return [
        { code: 'en', name: 'English' },
        { code: 'ur', name: 'Urdu' },
        { code: 'fr', name: 'French' },
        { code: 'de', name: 'German' },
        { code: 'es', name: 'Spanish' },
        { code: 'zh', name: 'Chinese' },
        { code: 'hi', name: 'Hindi' },
        { code: 'ja', name: 'Japanese' },
        { code: 'ko', name: 'Korean' }
      ];
    }
  }

  // Function to detect language using LibreTranslate API
  async detectLanguage(text) {
    try {
      const response = await fetch(`${this.libreTranslateUrl}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          q: text,
          api_key: this.apiKey || undefined
        })
      });

      if (!response.ok) {
        throw new Error(`Language detection API error: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();

      if (data[0] && data[0].language !== undefined) {
        return data[0].language;
      } else {
        throw new Error('Unexpected response format from language detection');
      }
    } catch (error) {
      console.error('Language detection error:', error);
      return 'en'; // Default to English
    }
  }
}

export default new TranslationService();