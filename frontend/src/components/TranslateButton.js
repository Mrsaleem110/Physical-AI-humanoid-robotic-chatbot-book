import React from 'react';
import { useTranslation } from '../contexts/TranslationContext';
import { useLocation, useHistory } from '@docusaurus/router';

const TranslateButton = () => {
  const { isTranslating, currentLanguage, supportedLanguages, translatePage, isRTL } = useTranslation();
  const location = useLocation();
  const history = useHistory();

  const handleLanguageChange = async (targetLanguage) => {
    if (targetLanguage !== currentLanguage) {
      await translatePage(targetLanguage);

      // Update URL to reflect new language
      const pathParts = location.pathname.split('/');
      let newPath;
      if (pathParts[1] && supportedLanguages.some(lang => lang.code === pathParts[1])) {
        newPath = `/${targetLanguage}` + pathParts.slice(2).join('/');
      } else {
        // If current path doesn't start with a language code, just use the target language
        newPath = location.pathname.startsWith(`/${targetLanguage}`) ?
          location.pathname :
          `/${targetLanguage}${location.pathname}`;
      }

      history.push(newPath);
    }
  };

  return (
    <div className="translate-button-container">
      <div className="dropdown dropdown--right dropdown--bottom">
        <button
          className={`button button--secondary ${isTranslating ? 'button--loading' : ''}`}
          disabled={isTranslating}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '8px 12px'
          }}
        >
          {isTranslating ? (
            <>
              <span className="loading-spinner">🔄</span>
              Translating...
            </>
          ) : (
            <>
              <span>🌐</span>
              Translate ({supportedLanguages.find(lang => lang.code === currentLanguage)?.name})
            </>
          )}
        </button>

        <ul className="dropdown__menu">
          {supportedLanguages
            .filter(lang => lang.code !== currentLanguage)
            .map(lang => (
              <li key={lang.code}>
                <a
                  className="dropdown__link"
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    handleLanguageChange(lang.code);
                  }}
                >
                  {lang.name}
                </a>
              </li>
            ))}
        </ul>
      </div>
    </div>
  );
};

export default TranslateButton;