import { useState, useEffect, useContext } from 'react';
import { TranslationContext } from '../components/Translation/TranslationProvider';

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