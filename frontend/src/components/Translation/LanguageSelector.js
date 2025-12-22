import React from 'react';
import { useTranslation } from '../../hooks/useTranslation';

// Language code to flag emoji mapping
const LANGUAGE_FLAGS = {
  'en': '🇺🇸', // English
  'ur': '🇵🇰', // Urdu
  'fr': '🇫🇷', // French
  'de': '🇩🇪', // German
  'es': '🇪🇸', // Spanish
  'zh': '🇨🇳', // Chinese
  'hi': '🇮🇳', // Hindi
  'ja': '🇯🇵', // Japanese
};

const LanguageSelector = () => {
  const {
    currentLanguage,
    availableLanguages,
    changeLanguage,
    isLoading,
    getLanguageName
  } = useTranslation();

  const handleLanguageChange = (event) => {
    const newLanguage = event.target.value;
    changeLanguage(newLanguage);
  };

  return (
    <div className="language-selector" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <label htmlFor="language-select" style={{ fontSize: '14px', cursor: 'pointer' }}>
        {LANGUAGE_FLAGS[currentLanguage] || '🌐'} {/* Show current language flag */}
      </label>
      <select
        id="language-select"
        value={currentLanguage}
        onChange={handleLanguageChange}
        disabled={isLoading}
        aria-label="Select language"
        style={{
          padding: '4px 8px',
          borderRadius: '4px',
          border: '1px solid #ccc',
          fontSize: '14px',
          backgroundColor: isLoading ? '#f5f5f5' : 'white',
          cursor: isLoading ? 'not-allowed' : 'pointer',
        }}
      >
        {availableLanguages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {LANGUAGE_FLAGS[lang.code] || ''} {lang.name} ({lang.code})
          </option>
        ))}
      </select>
      {isLoading && (
        <span
          style={{ marginLeft: '8px', fontSize: '12px', color: '#666' }}
          aria-live="polite"
          aria-label="Translation in progress"
        >
          Translating...
        </span>
      )}
    </div>
  );
};

export default LanguageSelector;