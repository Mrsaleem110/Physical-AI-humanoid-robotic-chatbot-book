import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import { translate } from '@docusaurus/Translate';

const TranslationButton = () => {
  const location = useLocation();
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [showLanguageMenu, setShowLanguageMenu] = useState(false);

  // Supported languages mapping
  const languageMap = {
    'en': { name: 'English', nativeName: 'English', flag: '🇬🇧' },
    'ur': { name: 'Urdu', nativeName: 'اردو', flag: '🇵🇰' },
    'hi': { name: 'Hindi', nativeName: 'हिंदी', flag: '🇮🇳' },
    'es': { name: 'Spanish', nativeName: 'Español', flag: '🇪🇸' },
    'ja': { name: 'Japanese', nativeName: '日本語', flag: '🇯🇵' },
    'zh': { name: 'Chinese', nativeName: '中文', flag: '🇨🇳' },
    'fr': { name: 'French', nativeName: 'Français', flag: '🇫🇷' },
    'de': { name: 'German', nativeName: 'Deutsch', flag: '🇩🇪' }
  };

  // Check current locale from URL
  useEffect(() => {
    const pathSegments = location.pathname.split('/');
    if (pathSegments.length > 1 && languageMap[pathSegments[1]]) {
      setCurrentLanguage(pathSegments[1]);
    } else {
      setCurrentLanguage('en');
    }
  }, [location]);

  const switchLanguage = (langCode) => {
    const currentPath = location.pathname;
    let newPath = currentPath;

    // Remove existing language prefix
    const existingLang = Object.keys(languageMap).find(code =>
      currentPath.startsWith(`/${code}/`) || currentPath === `/${code}`
    );

    if (existingLang) {
      newPath = currentPath.replace(`/${existingLang}`, '');
    } else if (currentPath.startsWith('/docs') || currentPath === '/' || currentPath.startsWith('/auth')) {
      // No language prefix found, keep the path as is
    } else {
      // For other paths, normalize them
      if (!currentPath.startsWith('/')) {
        newPath = '/' + currentPath;
      }
    }

    // Add new language prefix if not English (default)
    if (langCode === 'en') {
      // For English, remove language prefix and go to root
      if (newPath === '' || newPath === '/') {
        window.location.href = '/';
      } else {
        window.location.href = newPath;
      }
    } else {
      // For other languages, add the language prefix
      if (newPath === '' || newPath === '/') {
        window.location.href = `/${langCode}/`;
      } else {
        window.location.href = `/${langCode}${newPath}`;
      }
    }

    setShowLanguageMenu(false);
  };

  const toggleLanguageMenu = () => {
    setShowLanguageMenu(!showLanguageMenu);
  };

  const currentLangInfo = languageMap[currentLanguage] || languageMap['en'];

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={toggleLanguageMenu}
        style={{
          backgroundColor: '#007cba',
          color: 'white',
          border: 'none',
          padding: '8px 16px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '14px',
          fontWeight: '500',
          transition: 'background-color 0.3s ease',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}
        onMouseEnter={(e) => {
          e.target.style.backgroundColor = '#005a87';
        }}
        onMouseLeave={(e) => {
          e.target.style.backgroundColor = '#007cba';
        }}
      >
        {currentLangInfo.flag} {currentLangInfo.nativeName}
      </button>

      {showLanguageMenu && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: '0',
          backgroundColor: 'white',
          border: '1px solid #ddd',
          borderRadius: '4px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
          zIndex: 1000,
          minWidth: '150px',
          marginTop: '4px'
        }}>
          {Object.entries(languageMap).map(([code, lang]) => (
            <button
              key={code}
              onClick={() => switchLanguage(code)}
              style={{
                width: '100%',
                padding: '10px 12px',
                border: 'none',
                backgroundColor: currentLanguage === code ? '#f0f0f0' : 'white',
                textAlign: 'left',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '14px'
              }}
              onMouseEnter={(e) => {
                if (currentLanguage !== code) {
                  e.target.style.backgroundColor = '#f5f5f5';
                }
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = currentLanguage === code ? '#f0f0f0' : 'white';
              }}
            >
              {lang.flag} {lang.nativeName} ({lang.name})
            </button>
          ))}
        </div>
      )}

      {/* Close menu when clicking outside */}
      {showLanguageMenu && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: 999
          }}
          onClick={() => setShowLanguageMenu(false)}
        />
      )}
    </div>
  );
};

export default TranslationButton;