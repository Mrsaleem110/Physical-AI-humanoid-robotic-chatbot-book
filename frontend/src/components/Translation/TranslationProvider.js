import React, { useState, useEffect, createContext, useContext } from 'react';
import TranslationService from '../../services/translationService';
import { getLanguagePreference, setLanguagePreference } from '../../utils/storage';

// Create context for translation
const TranslationContext = createContext();

// Supported languages
const SUPPORTED_LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'ur', name: 'Urdu' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'es', name: 'Spanish' },
  { code: 'zh', name: 'Chinese' },
  { code: 'hi', name: 'Hindi' },
  { code: 'ja', name: 'Japanese' }
];

// Translation context provider component
export const TranslationProvider = ({ children, defaultLanguage = 'en' }) => {
  const [currentLanguage, setCurrentLanguage] = useState(defaultLanguage);
  const [isLoading, setIsLoading] = useState(false);
  const [availableLanguages, setAvailableLanguages] = useState(SUPPORTED_LANGUAGES);

  // Load saved language preference on initialization
  useEffect(() => {
    const savedLanguage = getLanguagePreference();
    setCurrentLanguage(savedLanguage);
  }, []);

  // Function to change language
  const changeLanguage = async (newLanguage) => {
    if (!SUPPORTED_LANGUAGES.some(lang => lang.code === newLanguage)) {
      console.error(`Language ${newLanguage} is not supported`);
      return;
    }

    setIsLoading(true);
    try {
      setCurrentLanguage(newLanguage);
      setLanguagePreference(newLanguage);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to translate text
  const translate = async (text, targetLang = currentLanguage, sourceLang = 'auto') => {
    if (!text || typeof text !== 'string') {
      return text;
    }

    if (targetLang === 'en') {
      return text; // No translation needed if target is English
    }

    try {
      const result = await TranslationService.translateText(text, targetLang, sourceLang);
      return result;
    } catch (error) {
      console.error('Translation error:', error);
      return text; // Return original text if translation fails - fallback to English requirement
    }
  };

  // Get language name by code
  const getLanguageName = (code) => {
    const lang = SUPPORTED_LANGUAGES.find(l => l.code === code);
    return lang ? lang.name : code;
  };

  const value = {
    currentLanguage,
    availableLanguages,
    isLoading,
    changeLanguage,
    translate,
    getLanguageName
  };

  return (
    <TranslationContext.Provider value={value}>
      {children}
    </TranslationContext.Provider>
  );
};

// Custom hook to use translation context
export const useTranslation = () => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslation must be used within a TranslationProvider');
  }
  return context;
};

// Hook to get translation for a specific text
export const useTranslateText = (text) => {
  const { currentLanguage, translate, isLoading } = useTranslation();
  const [translatedText, setTranslatedText] = useState(text);

  useEffect(() => {
    const fetchTranslation = async () => {
      if (currentLanguage !== 'en' && text) {
        const result = await translate(text, currentLanguage);
        setTranslatedText(result);
      } else {
        setTranslatedText(text);
      }
    };

    fetchTranslation();
  }, [text, currentLanguage, translate]);

  return { translatedText, isLoading };
};