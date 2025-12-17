import React, { useState } from 'react';
import { useTranslation } from '../contexts/TranslationContext';

// Component to display translated text
const TranslatedText = ({ children, sourceLang = 'en' }) => {
  const { currentLanguage, isTranslating } = useTranslation();
  const [translatedText, setTranslatedText] = useState(children);

  // In a real implementation, you would translate the text
  // For now, we'll just return the original text
  // The actual translation happens at the page level through the context
  return (
    <span
      data-translate={children}
      data-source-lang={sourceLang}
      data-target-lang={currentLanguage}
    >
      {children}
    </span>
  );
};

// Higher-order component to wrap content that should be translatable
export const withTranslation = (WrappedComponent) => {
  return (props) => {
    const { currentLanguage } = useTranslation();

    return (
      <div
        className={`translatable-content ${props.className || ''}`}
        data-current-lang={currentLanguage}
      >
        <WrappedComponent {...props} />
      </div>
    );
  };
};

// Hook to translate individual text strings
export const useTranslate = () => {
  const { currentLanguage } = useTranslation();

  const translate = (text, sourceLang = 'en') => {
    // In a real implementation, this would call the translation service
    // For now, we return the text as-is since page-level translation handles it
    return text;
  };

  return { translate, currentLanguage };
};

export default TranslatedText;