import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from '@docusaurus/router';

const MultilingualDropdown = () => {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Supported languages with their native names and codes
  const languages = [
    { code: 'en', name: 'English', native: 'English' },
    { code: 'ur', name: 'Urdu', native: 'اردو' }
  ];

  // Get current language from URL
  const getCurrentLanguage = () => {
    const pathParts = location.pathname.split('/');
    if (pathParts[1] && languages.some(lang => lang.code === pathParts[1])) {
      return pathParts[1];
    }
    return 'en'; // default to English
  };

  const currentLang = getCurrentLanguage();
  const currentLanguageInfo = languages.find(lang => lang.code === currentLang);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLanguageChange = (langCode) => {
    const currentPath = location.pathname;
    const currentHost = window.location.origin;
    const currentLang = getCurrentLanguage();

    if (langCode === 'en') {
      // For English, remove language prefix or go to root
      const newPath = currentPath.replace(/^\/[^\/]+\//, '/');
      window.location.href = currentHost + (newPath === '/' ? '/' : newPath);
    } else {
      // For other languages, add the language prefix
      let newPath = currentPath;
      // Remove existing language prefix if present
      const pathParts = currentPath.split('/');
      if (pathParts[1] && languages.some(lang => lang.code === pathParts[1])) {
        newPath = '/' + pathParts.slice(2).join('/');
      }
      if (newPath === '/' || newPath === '') {
        window.location.href = currentHost + '/' + langCode + '/';
      } else {
        window.location.href = currentHost + '/' + langCode + newPath;
      }
    }
    setIsOpen(false);
  };

  return (
    <div className="multilingual-dropdown" ref={dropdownRef} style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          backgroundColor: '#007cba',
          color: 'white',
          border: 'none',
          padding: '8px 12px',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '14px',
          fontWeight: '500',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          transition: 'background-color 0.3s ease',
        }}
        onMouseEnter={(e) => {
          e.target.style.backgroundColor = '#005a87';
        }}
        onMouseLeave={(e) => {
          e.target.style.backgroundColor = '#007cba';
        }}
      >
        <span>🌐</span>
        <span>{currentLanguageInfo?.native || 'English'}</span>
        <span style={{ marginLeft: '4px' }}>{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: '0',
          backgroundColor: 'white',
          border: '1px solid #ddd',
          borderRadius: '4px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
          zIndex: '1000',
          minWidth: '150px',
          marginTop: '4px',
        }}>
          {languages.map((lang) => (
            <div
              key={lang.code}
              onClick={() => handleLanguageChange(lang.code)}
              style={{
                padding: '8px 12px',
                cursor: 'pointer',
                borderBottom: lang.code !== languages[languages.length - 1].code ? '1px solid #eee' : 'none',
                backgroundColor: currentLang === lang.code ? '#f5f5f5' : 'white',
                fontWeight: currentLang === lang.code ? 'bold' : 'normal',
              }}
              onMouseEnter={(e) => {
                if (currentLang !== lang.code) {
                  e.target.style.backgroundColor = '#f0f0f0';
                }
              }}
              onMouseLeave={(e) => {
                if (currentLang !== lang.code) {
                  e.target.style.backgroundColor = 'white';
                }
              }}
            >
              {lang.native} ({lang.name})
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MultilingualDropdown;