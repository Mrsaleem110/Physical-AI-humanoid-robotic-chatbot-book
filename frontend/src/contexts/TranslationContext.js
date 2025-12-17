import React, { createContext, useContext, useState, useEffect } from 'react';

// Translation service to handle all translation functionality
class TranslationService {
  constructor() {
    // Safe environment variable access for Docusaurus
    this.apiUrl = this.getApiUrl();
    this.cache = new Map();
  }

  getApiUrl() {
    // Check if we're in the browser environment first
    if (typeof window !== 'undefined') {
      // In browser environment - use backend API
      return '/api/translation/translate-english-to-urdu'; // Default to English to Urdu endpoint
    } else {
      // In Node.js environment (SSR) - need to check if process exists
      if (typeof process !== 'undefined' && process.env) {
        // Use the backend API URL from environment variables
        return process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
      } else {
        return 'http://localhost:8000';
      }
    }
  }

  getApiKey() {
    // For backend API, we might use a different auth method
    if (typeof window !== 'undefined' && window.localStorage) {
      return localStorage.getItem('access_token'); // Use stored auth token
    }
    return undefined;
  }

  async translateText(text, sourceLang = 'en', targetLang = 'ur') {
    // Skip if text is empty or too short
    if (!text || text.trim().length === 0) {
      return text;
    }

    const cacheKey = `${sourceLang}-${targetLang}-${text.substring(0, 50)}`;

    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    try {
      // Use the backend translation service instead of LibreTranslate
      let apiUrl, requestBody, headers;

      // Determine which backend endpoint to use based on source and target languages
      // Using the public translation endpoints that don't require authentication
      apiUrl = `${process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000'}/api/translate/translate`;
      requestBody = JSON.stringify({
        content: text,
        target_language: targetLang,
        source_language: sourceLang,
        preserve_formatting: false
      });

      headers = {
        'Content-Type': 'application/json',
      };

      // Add auth token if available
      const authToken = this.getApiKey();
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: requestBody,
      });

      if (!response.ok) {
        console.error(`Backend translation API error: ${response.status} ${response.statusText}`);
        throw new Error(`Backend translation API error: ${response.status}`);
      }

      const data = await response.json();
      const translatedText = data.translated_content || data.translatedText || data.original_content || data.original_text || text;

      // Cache the result
      this.cache.set(cacheKey, translatedText);

      return translatedText;
    } catch (error) {
      console.error('Translation error:', error);
      // Fallback to mock translation to avoid breaking the UI
      if (targetLang === 'ur') {
        return `[URDU: ${text}]`; // Simple mock for Urdu
      } else if (targetLang === 'en') {
        return `[ENGLISH: ${text}]`; // Simple mock for English
      } else {
        return text; // Return original text if translation fails
      }
    }
  }

  async translatePageContent(targetLanguage, sourceLanguage = 'en') {
    // Get all translatable elements, avoiding code blocks and other non-text elements
    const elements = document.querySelectorAll(
      'h1:not(.code), h2:not(.code), h3:not(.code), h4:not(.code), h5:not(.code), h6:not(.code), ' +
      'p:not(.code), span:not(.code):not([class*="token"]):not([class*="keyword"]):not([class*="operator"]), ' +
      'div:not(.code):not([class*="token"]):not([class*="keyword"]):not([class*="operator"]), ' +
      'li:not(.code), td:not(.code), th:not(.code), .markdown p, .markdown h1, .markdown h2, .markdown h3, ' +
      '.markdown h4, .markdown h5, .markdown h6, .markdown span, .markdown div, .markdown li, ' +
      '.hero__title, .hero__subtitle, .container, .theme-doc-markdown'
    );

    const translationPromises = [];

    elements.forEach(element => {
      // Skip if element is inside a code block or already translated
      if (element.closest('code, pre') || element.hasAttribute('data-translated')) {
        return;
      }

      // Get text content, but only if it's primarily text (not a container with many nested elements)
      const originalText = element.textContent.trim();

      // Skip if element is empty, too short, or contains mainly special characters
      if (originalText &&
          originalText.length > 2 &&
          !/^[\s\n\r]*$/.test(originalText) &&
          !/^[\s\n\r0-9\W]*$/.test(originalText)) {

        const promise = this.translateText(originalText, sourceLanguage, targetLanguage)
          .then(translatedText => {
            // Store original text for potential reversion
            element.setAttribute('data-original-text', originalText);
            element.setAttribute('data-translated', 'true');
            element.textContent = translatedText;
          });

        translationPromises.push(promise);
      }
    });

    // Wait for all translations to complete
    await Promise.all(translationPromises);
  }

  // Method to revert translations back to original text
  revertTranslations() {
    const translatedElements = document.querySelectorAll('[data-translated="true"]');
    translatedElements.forEach(element => {
      const originalText = element.getAttribute('data-original-text');
      if (originalText) {
        element.textContent = originalText;
      }
      element.removeAttribute('data-translated');
      element.removeAttribute('data-original-text');
    });
  }

  // Method to re-translate current page content to maintain language
  async retranslateCurrentPage(targetLanguage, sourceLanguage = 'en') {
    // Wait a bit for page to fully load
    await new Promise(resolve => setTimeout(resolve, 300));

    // Find all elements that have been translated and re-translate them
    const translatedElements = document.querySelectorAll('[data-translated="true"]');

    const translationPromises = [];

    translatedElements.forEach(element => {
      const originalText = element.getAttribute('data-original-text');
      if (originalText) {
        const promise = this.translateText(originalText, sourceLanguage, targetLanguage)
          .then(translatedText => {
            element.textContent = translatedText;
          });

        translationPromises.push(promise);
      }
    });

    await Promise.all(translationPromises);
  }
}

export const TranslationContext = createContext();

export const TranslationProvider = ({ children }) => {
  const [translationService] = useState(() => new TranslationService());
  const [isTranslating, setIsTranslating] = useState(false);
  const [currentLanguage, setCurrentLanguage] = useState('en');

  // Function to get the current language from URL (for Docusaurus i18n)
  const getCurrentDocusaurusLanguage = () => {
    if (typeof window !== 'undefined') {
      const path = window.location.pathname;
      const pathParts = path.split('/');
      if (pathParts[1] && pathParts[1].length === 2) { // Simple check for 2-letter language codes
        const possibleLang = pathParts[1].toLowerCase();
        const supportedLangs = ['en', 'ur'];
        if (supportedLangs.includes(possibleLang)) {
          return possibleLang;
        }
      }
    }
    return 'en'; // default to English
  };

  // Initialize with the current Docusaurus language
  useEffect(() => {
    const docusaurusLang = getCurrentDocusaurusLanguage();
    setCurrentLanguage(docusaurusLang);
  }, []);

  // Listen for route changes to re-apply translations if needed
  useEffect(() => {
    let timeoutId;

    const handleRouteChange = () => {
      // Re-apply translations when navigating to a new page if we're in a translated state
      if (currentLanguage !== 'en') {
        // Use a slight delay to ensure the new page content is loaded
        timeoutId = setTimeout(() => {
          if (translationService && currentLanguage) {
            translationService.retranslateCurrentPage(currentLanguage);
          }
        }, 500);
      }
    };

    // Docusaurus uses history API for client-side routing
    if (typeof window !== 'undefined') {
      // Listen for popstate events (browser back/forward)
      window.addEventListener('popstate', handleRouteChange);

      // For Docusaurus-specific events, we'll rely on DOM changes
      const observer = new MutationObserver((mutations) => {
        // Check if page content has changed significantly (indicating navigation)
        for (let mutation of mutations) {
          if (mutation.type === 'childList') {
            // Look for changes that indicate a page navigation
            const addedNodes = Array.from(mutation.addedNodes);
            if (addedNodes.some(node =>
              node.nodeType === 1 &&
              (node.classList.contains('main-wrapper') ||
               node.classList.contains('container') ||
               node.classList.contains('docPage') ||
               node.classList.contains('theme-doc-markdown'))
            )) {
              handleRouteChange();
            }
          }
        }
      });

      observer.observe(document.body, {
        childList: true,
        subtree: true
      });

      return () => {
        if (timeoutId) clearTimeout(timeoutId);
        window.removeEventListener('popstate', handleRouteChange);
        observer.disconnect();
      };
    }
  }, [currentLanguage, translationService]);

  const translatePage = async (targetLanguage) => {
    if (targetLanguage === currentLanguage) return;

    setIsTranslating(true);
    try {
      // Revert previous translations first
      translationService.revertTranslations();
      // Translate to new language
      await translationService.translatePageContent(targetLanguage, currentLanguage);
      setCurrentLanguage(targetLanguage);
    } catch (error) {
      console.error('Translation failed:', error);
    } finally {
      setIsTranslating(false);
    }
  };

  const value = {
    translatePage,
    isTranslating,
    currentLanguage,
    supportedLanguages: [
      { code: 'en', name: 'English' },
      { code: 'ur', name: 'Urdu' }
    ],
  };

  return (
    <TranslationContext.Provider value={value}>
      {children}
    </TranslationContext.Provider>
  );
};

export const useTranslation = () => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
};