import React, { useEffect, useState } from 'react';
import { useTranslation } from '../../hooks/useTranslation';

const TranslateText = ({ children, fallback = null }) => {
  const { currentLanguage, translate, isLoading } = useTranslation();
  const [translatedText, setTranslatedText] = useState(children);

  useEffect(() => {
    const translateContent = async () => {
      if (currentLanguage !== 'en' && children && typeof children === 'string') {
        try {
          const result = await translate(children, currentLanguage);
          setTranslatedText(result);
        } catch (error) {
          console.error('Translation error:', error);
          setTranslatedText(children); // Fallback to original text
        }
      } else {
        setTranslatedText(children);
      }
    };

    if (children && typeof children === 'string') {
      translateContent();
    } else {
      setTranslatedText(children);
    }
  }, [children, currentLanguage, translate]);

  if (isLoading && typeof children === 'string') {
    return <span>...</span>; // Show loading indicator for text content
  }

  return <>{translatedText}</>;
};

export default TranslateText;