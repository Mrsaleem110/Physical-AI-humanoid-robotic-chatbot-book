import React, { useState, useEffect } from 'react';

// Simple fallback component when translation context is not available
const TranslationFallback = () => {
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [isOpen, setIsOpen] = useState(false);

  const languageMap = {
    'en': { name: 'English', native: 'English', flag: '🇬🇧' },
    'ur': { name: 'Urdu', native: 'اردو', flag: '🇵🇰', direction: 'rtl' }
  };

  const currentLangInfo = languageMap[currentLanguage] || languageMap['en'];

  const handleLanguageChange = (langCode) => {
    if (langCode === currentLanguage) {
      setIsOpen(false);
      return;
    }

    // For fallback, just update the language state and document attributes
    setCurrentLanguage(langCode);
    document.documentElement.lang = langCode;
    if (languageMap[langCode]?.direction === 'rtl') {
      document.documentElement.dir = 'rtl';
    } else {
      document.documentElement.dir = 'ltr';
    }

    setIsOpen(false);

    // Show a message to the user that full translation requires login or setup
    alert(`Language changed to ${languageMap[langCode].native}. Full translation requires proper setup.`);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      const dropdown = document.getElementById('complete-translation-dropdown');
      if (dropdown && !dropdown.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div id="complete-translation-dropdown" style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          backgroundColor: '#6c757d',
          color: 'white',
          border: 'none',
          padding: '10px 15px',
          borderRadius: '6px',
          cursor: 'pointer',
          fontSize: '14px',
          fontWeight: '500',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          transition: 'all 0.3s ease',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          minWidth: '140px',
          justifyContent: 'space-between'
        }}
        onMouseEnter={(e) => {
          e.target.style.backgroundColor = '#5a6268';
          e.target.style.transform = 'translateY(-1px)';
        }}
        onMouseLeave={(e) => {
          e.target.style.backgroundColor = '#6c757d';
          e.target.style.transform = 'translateY(0)';
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span>{currentLangInfo.flag}</span>
          <span>{currentLangInfo.native}</span>
        </span>
        <span style={{ fontSize: '12px' }}>{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: '0',
          backgroundColor: 'white',
          border: '1px solid #ddd',
          borderRadius: '6px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          zIndex: '1000',
          minWidth: '180px',
          marginTop: '4px',
          maxHeight: '300px',
          overflowY: 'auto'
        }}>
          {Object.entries(languageMap).map(([code, info]) => (
            <div
              key={code}
              onClick={() => handleLanguageChange(code)}
              style={{
                padding: '12px 16px',
                cursor: 'pointer',
                borderBottom: code !== Object.keys(languageMap)[Object.keys(languageMap).length - 1] ? '1px solid #eee' : 'none',
                backgroundColor: currentLanguage === code ? '#f0f8ff' : 'white',
                fontWeight: currentLanguage === code ? 'bold' : 'normal',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'background-color 0.2s ease'
              }}
              onMouseEnter={(e) => {
                if (currentLanguage !== code) {
                  e.target.style.backgroundColor = '#f8f9fa';
                }
              }}
              onMouseLeave={(e) => {
                if (currentLanguage !== code) {
                  e.target.style.backgroundColor = 'white';
                }
              }}
            >
              <span>{info.flag}</span>
              <span>{info.native}</span>
              <span style={{ color: '#666', fontSize: '12px', marginLeft: 'auto' }}>
                {info.name}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Main CompleteWebsiteTranslation component that handles client-side rendering properly
const CompleteWebsiteTranslation = () => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    // Render nothing during SSR, or a simple placeholder
    return (
      <div style={{
        padding: '10px 15px',
        backgroundColor: '#6c757d',
        color: 'white',
        borderRadius: '6px',
        fontSize: '14px'
      }}>
        🌐
      </div>
    );
  }

          // Check if we're in the browser environment before accessing context
    if (typeof window === 'undefined') {
      return <TranslationFallback />;
    }

    try {
      // Import the context (this should work after component has mounted)
      const contextModule = require('../contexts/TranslationContext');
      const { useTranslation } = contextModule;

      // Use the TranslationContext directly
      const { currentLanguage, translatePage } = useTranslation();
      const [isOpen, setIsOpen] = useState(false);
      const [showConfirmation, setShowConfirmation] = useState(false);

      const languageMap = {
        'en': { name: 'English', native: 'English', flag: '🇬🇧' },
        'ur': { name: 'Urdu', native: 'اردو', flag: '🇵🇰', direction: 'rtl' }
      };

      const currentLangInfo = languageMap[currentLanguage] || languageMap['en'];

      const handleLanguageChange = async (langCode) => {
        if (langCode === currentLanguage) {
          setIsOpen(false);
          return;
        }

        if (translatePage) {
          await translatePage(langCode);

          // Update document attributes for proper language and direction
          document.documentElement.lang = langCode;
          if (languageMap[langCode]?.direction === 'rtl') {
            document.documentElement.dir = 'rtl';
          } else {
            document.documentElement.dir = 'ltr';
          }
        }

        setIsOpen(false);
        setShowConfirmation(true);
        setTimeout(() => setShowConfirmation(false), 3000);
      };

      // Close dropdown when clicking outside
      useEffect(() => {
        const handleClickOutside = (event) => {
          const dropdown = document.getElementById('complete-translation-dropdown');
          if (dropdown && !dropdown.contains(event.target)) {
            setIsOpen(false);
          }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
      }, []);

      return (
        <div id="complete-translation-dropdown" style={{ position: 'relative', display: 'inline-block' }}>
          <button
            onClick={() => setIsOpen(!isOpen)}
            style={{
              backgroundColor: '#007cba',
              color: 'white',
              border: 'none',
              padding: '10px 15px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.3s ease',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              minWidth: '140px',
              justifyContent: 'space-between'
            }}
            onMouseEnter={(e) => {
              e.target.style.backgroundColor = '#005a87';
              e.target.style.transform = 'translateY(-1px)';
            }}
            onMouseLeave={(e) => {
              e.target.style.backgroundColor = '#007cba';
              e.target.style.transform = 'translateY(0)';
            }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>{currentLangInfo.flag}</span>
              <span>{currentLangInfo.native}</span>
            </span>
            <span style={{ fontSize: '12px' }}>{isOpen ? '▲' : '▼'}</span>
          </button>

          {isOpen && (
            <div style={{
              position: 'absolute',
              top: '100%',
              right: '0',
              backgroundColor: 'white',
              border: '1px solid #ddd',
              borderRadius: '6px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: '1000',
              minWidth: '180px',
              marginTop: '4px',
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
              {Object.entries(languageMap).map(([code, info]) => (
                <div
                  key={code}
                  onClick={() => handleLanguageChange(code)}
                  style={{
                    padding: '12px 16px',
                    cursor: 'pointer',
                    borderBottom: code !== Object.keys(languageMap)[Object.keys(languageMap).length - 1] ? '1px solid #eee' : 'none',
                    backgroundColor: currentLanguage === code ? '#f0f8ff' : 'white',
                    fontWeight: currentLanguage === code ? 'bold' : 'normal',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    transition: 'background-color 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (currentLanguage !== code) {
                      e.target.style.backgroundColor = '#f8f9fa';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (currentLanguage !== code) {
                      e.target.style.backgroundColor = 'white';
                    }
                  }}
                >
                  <span>{info.flag}</span>
                  <span>{info.native}</span>
                  <span style={{ color: '#666', fontSize: '12px', marginLeft: 'auto' }}>
                    {info.name}
                  </span>
                </div>
              ))}
            </div>
          )}

          {showConfirmation && (
            <div style={{
              position: 'fixed',
              top: '20px',
              right: '20px',
              backgroundColor: '#28a745',
              color: 'white',
              padding: '12px 20px',
              borderRadius: '6px',
              boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
              zIndex: '10000',
              fontSize: '14px',
              fontWeight: '500',
              animation: 'slideIn 0.3s ease, fadeOut 0.5s ease 2.5s forwards'
            }}>
              🌐 Website translated to {currentLangInfo.native} ({currentLangInfo.name})!
            </div>
          )}

          <style jsx>{`
            @keyframes slideIn {
              from {
                transform: translateX(100%);
                opacity: 0;
              }
              to {
                transform: translateX(0);
                opacity: 1;
              }
            }
            @keyframes fadeOut {
              from {
                opacity: 1;
              }
              to {
                opacity: 0;
              }
            }
          `}</style>
        </div>
      );
    } catch (error) {
      // If context is not available, return the fallback component
      console.error('Translation context error:', error);
      return <TranslationFallback />;
    }
};

export default CompleteWebsiteTranslation;