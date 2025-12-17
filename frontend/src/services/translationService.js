// Translation Service for Page Translation Component
// This service handles translation of page content using various translation APIs

class TranslationService {
  constructor() {
    // You can configure your translation API key here
    // For example, if using Google Translate API:
    this.apiKey = process.env.REACT_APP_TRANSLATE_API_KEY || null; // Check for environment variable
    this.googleApiKey = process.env.REACT_APP_GOOGLE_TRANSLATE_API_KEY || this.apiKey;
    this.azureApiKey = process.env.REACT_APP_AZURE_TRANSLATOR_KEY || null;
    this.azureRegion = process.env.REACT_APP_AZURE_TRANSLATOR_REGION || 'global';
  }

  // Function to translate text using Google Translate API
  async translateTextWithGoogle(text, targetLang, sourceLang = 'en') {
    if (!this.googleApiKey) {
      throw new Error('Google Translate API key not configured');
    }

    const response = await fetch(`https://translation.googleapis.com/language/translate/v2`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.googleApiKey}`,
      },
      body: JSON.stringify({
        q: text,
        target: targetLang,
        source: sourceLang,
        format: 'text'
      })
    });

    if (!response.ok) {
      throw new Error(`Google Translate API error: ${response.status}`);
    }

    const data = await response.json();
    return data.data.translations[0].translatedText;
  }

  // Function to translate text using Azure Translator
  async translateTextWithAzure(text, targetLang, sourceLang = 'en') {
    if (!this.azureApiKey) {
      throw new Error('Azure Translator API key not configured');
    }

    const response = await fetch(`https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=${targetLang}&from=${sourceLang}`, {
      method: 'POST',
      headers: {
        'Ocp-Apim-Subscription-Key': this.azureApiKey,
        'Ocp-Apim-Subscription-Region': this.azureRegion,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify([{
        Text: text
      }])
    });

    if (!response.ok) {
      throw new Error(`Azure Translator API error: ${response.status}`);
    }

    const data = await response.json();
    return data[0].translations[0].text;
  }

  // Function to translate text using a translation API (with fallback)
  async translateText(text, targetLang, sourceLang = 'en') {
    if (!text || text.trim().length === 0) {
      return text;
    }

    // Try LibreTranslate first if configured (since it's free and often available)
    if (process.env.REACT_APP_LIBRETRANSLATE_URL) {
      try {
        return await this.translateTextWithFreeService(text, targetLang, sourceLang);
      } catch (error) {
        console.error('LibreTranslate failed:', error);
        // Fall through to try other services
      }
    }

    // Try Google Translate first
    if (this.googleApiKey) {
      try {
        return await this.translateTextWithGoogle(text, targetLang, sourceLang);
      } catch (error) {
        console.error('Google Translate failed:', error);
        // Fall through to try Azure or mock
      }
    }

    // Try Azure Translator if Google failed
    if (this.azureApiKey) {
      try {
        return await this.translateTextWithAzure(text, targetLang, sourceLang);
      } catch (error) {
        console.error('Azure Translator failed:', error);
        // Fall through to mock translation
      }
    }

    // If all APIs fail, return mock translation
    console.warn('All translation APIs failed, using mock translation');
    return this.mockTranslate(text, targetLang);
  }

  // Function to translate text using a free service (example with LibreTranslate API)
  async translateTextWithFreeService(text, targetLang, sourceLang = 'en') {
    // This is an example using a public LibreTranslate instance
    // You can set up your own LibreTranslate instance or use a public one
    const libreTranslateUrl = process.env.REACT_APP_LIBRETRANSLATE_URL || 'https://libretranslate.com/translate';

    const response = await fetch(libreTranslateUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        q: text,
        source: sourceLang,
        target: targetLang,
        format: 'text'
      })
    });

    if (!response.ok) {
      throw new Error(`Free translation service error: ${response.status} - ${response.statusText}`);
    }

    const data = await response.json();

    // LibreTranslate may return different response formats depending on the instance
    // Handle both standard and alternative response formats
    if (data.translatedText !== undefined) {
      return data.translatedText;
    } else if (data[0] && data[0].translatedText !== undefined) {
      return data[0].translatedText;
    } else {
      throw new Error('Unexpected response format from LibreTranslate service');
    }
  }

  // Mock translation function for demonstration
  mockTranslate(text, targetLang) {
    // This is just for demonstration - in a real app you'd use a real API
    const langMap = {
      'ur': 'مترجم',
      'es': 'TRADUCIDO',
      'fr': 'TRADUIT',
      'de': 'ÜBERSETZT',
      'zh': '已翻译',
      'ja': '翻訳済み',
      'ko': '번역됨',
      'ar': 'مترجم',
      'hi': 'अनुवादित',
      'ru': 'ПЕРЕВЕДЕНО'
    };

    // For demonstration, we'll just add the language code as a prefix/suffix
    // In a real implementation, you would use an actual translation API
    const prefix = langMap[targetLang] || 'TRANSLATED';
    return `[${prefix}] ${text} [${targetLang.toUpperCase()}]`;
  }

  // Function to translate multiple texts in batch
  async translateBatch(texts, targetLang, sourceLang = 'en') {
    const results = {};

    // Process translations in batches to avoid overwhelming the API
    const batchSize = 5; // Adjust based on API limits
    const entries = Object.entries(texts);

    for (let i = 0; i < entries.length; i += batchSize) {
      const batch = entries.slice(i, i + batchSize);

      const batchPromises = batch.map(async ([key, text]) => {
        try {
          const translated = await this.translateText(text, targetLang, sourceLang);
          results[key] = translated;
        } catch (error) {
          console.error(`Translation error for key ${key}:`, error);
          results[key] = text; // Return original text on error
        }
      });

      await Promise.all(batchPromises);

      // Small delay between batches to respect API rate limits
      if (i + batchSize < entries.length) {
        await new Promise(resolve => setTimeout(resolve, 200)); // 200ms delay between batches
      }
    }

    return results;
  }

  // Function to get supported languages
  getSupportedLanguages() {
    return [
      { code: 'en', name: 'English' },
      { code: 'ur', name: 'Urdu' }
    ];
  }

  // Function to detect language
  async detectLanguage(text) {
    if (!this.googleApiKey) {
      throw new Error('Google Translate API key required for language detection');
    }

    const response = await fetch(`https://translation.googleapis.com/language/translate/v2/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.googleApiKey}`,
      },
      body: JSON.stringify({
        q: text
      })
    });

    if (!response.ok) {
      throw new Error(`Language detection API error: ${response.status}`);
    }

    const data = await response.json();
    return data.data.detections[0][0].language;
  }
}

export default new TranslationService();