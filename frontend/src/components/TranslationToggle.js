import React, { useState } from 'react';
import { useTranslation } from '../contexts/TranslationContext';

const TranslationToggle = () => {
  const { currentLanguage, translatePage, isTranslating, supportedLanguages } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  // Get current language info
  const currentLangInfo = supportedLanguages.find(lang => lang.code === currentLanguage) || { code: currentLanguage, name: currentLanguage, native: currentLanguage };

  // Handle language selection
  const handleLanguageSelect = (langCode) => {
    if (langCode !== currentLanguage) {
      translatePage(langCode);
    }
    setIsOpen(false);
  };

  // Get flag based on language
  const getFlagIcon = (langCode) => {
    switch (langCode) {
      case 'en': return '🇺🇸';
      case 'ur': return '🇵🇰';
      case 'ko': return '🇰🇷';
      case 'fr': return '🇫🇷';
      case 'de': return '🇩🇪';
      case 'es': return '🇪🇸';
      case 'zh': return '🇨🇳';
      case 'ja': return '🇯🇵';
      case 'hi': return '🇮🇳';
      default: return '🌐';
    }
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isTranslating}
        style={{
          background: isTranslating ? '#f0f0f0' : '#ffffff',
          border: '1px solid #ddd',
          borderRadius: '6px',
          padding: '6px 12px',
          cursor: isTranslating ? 'not-allowed' : 'pointer',
          fontSize: '14px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          transition: 'all 0.2s ease',
          boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
        }}
        onMouseOver={(e) => {
          if (!isTranslating) {
            e.target.style.backgroundColor = '#f8f9fa';
            e.target.style.borderColor = '#94a3b8';
          }
        }}
        onMouseOut={(e) => {
          e.target.style.backgroundColor = isTranslating ? '#f0f0f0' : '#ffffff';
          e.target.style.borderColor = '#ddd';
        }}
        title="Change language"
        aria-label="Change language"
        aria-expanded={isOpen}
      >
        <span role="img" aria-label={`${currentLangInfo.name} flag`}>
          {getFlagIcon(currentLanguage)}
        </span>
        <span style={{fontWeight: '500', color: '#334155'}}>
          {currentLangInfo.native}
        </span>
        <span style={{ fontSize: '10px' }}>
          {isOpen ? '▲' : '▼'}
        </span>
        {isTranslating && (
          <span style={{marginLeft: '4px'}}>🔄</span>
        )}
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            right: 0,
            background: '#ffffff',
            border: '1px solid #ddd',
            borderRadius: '6px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
            zIndex: 1000,
            minWidth: '140px',
            marginTop: '4px',
          }}
        >
          {supportedLanguages.map((lang) => (
            <button
              key={lang.code}
              onClick={() => handleLanguageSelect(lang.code)}
              style={{
                width: '100%',
                padding: '8px 12px',
                border: 'none',
                background: currentLanguage === lang.code ? '#f8f9fa' : '#ffffff',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                textAlign: 'left',
                fontSize: '14px',
                color: currentLanguage === lang.code ? '#1f2937' : '#374151',
                fontWeight: currentLanguage === lang.code ? '600' : '400',
              }}
              onMouseOver={(e) => {
                e.target.style.backgroundColor = '#f3f4f6';
              }}
              onMouseOut={(e) => {
                e.target.style.backgroundColor = currentLanguage === lang.code ? '#f8f9fa' : '#ffffff';
              }}
            >
              <span role="img" aria-label={`${lang.name} flag`}>
                {getFlagIcon(lang.code)}
              </span>
              <span>{lang.native}</span>
            </button>
          ))}
        </div>
      )}

      {/* Close dropdown when clicking outside */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            zIndex: 999,
            cursor: 'default',
          }}
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
};

export default TranslationToggle;