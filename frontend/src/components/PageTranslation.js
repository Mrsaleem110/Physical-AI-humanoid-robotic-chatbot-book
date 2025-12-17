import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import translationService from '../services/translationService';

const PageTranslation = () => {
  const location = useLocation();
  const [isTranslating, setIsTranslating] = useState(false);
  const [isTranslated, setIsTranslated] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('ur');
  const [originalContent, setOriginalContent] = useState({});

  const languageOptions = [
    { code: 'ur', name: 'Urdu' }
  ];

  // Function to get all text content from the page
  const extractPageContent = () => {
    const content = {};
    const textElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, td, th, div, span, a, button');

    textElements.forEach((element, index) => {
      const text = element.textContent.trim();
      if (text && !content[element.tagName + '_' + index]) {
        content[element.tagName + '_' + index] = text;
      }
    });

    return content;
  };


  // Function to apply translations to the page
  const applyTranslations = async (translations) => {
    const textElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, td, th, div, span, a, button');

    textElements.forEach((element, index) => {
      const key = element.tagName + '_' + index;
      if (translations[key]) {
        element.setAttribute('data-original-text', element.textContent);
        element.textContent = translations[key];
      }
    });
  };

  // Function to restore original content
  const restoreOriginalContent = () => {
    const textElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, li, td, th, div, span, a, button');

    textElements.forEach(element => {
      const originalText = element.getAttribute('data-original-text');
      if (originalText) {
        element.textContent = originalText;
        element.removeAttribute('data-original-text');
      }
    });
  };

  const handleTranslatePage = async () => {
    if (isTranslating) return;

    setIsTranslating(true);

    try {
      if (isTranslated) {
        // Restore original content
        restoreOriginalContent();
        setIsTranslated(false);
      } else {
        // Extract current page content
        const content = extractPageContent();
        setOriginalContent(content);

        // Translate all content using the translation service
        const translations = await translationService.translateBatch(content, targetLanguage);

        // Apply translations to the page
        applyTranslations(translations);
        setIsTranslated(true);
      }
    } catch (error) {
      console.error('Translation error:', error);
      alert('Translation failed. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  // Reset translation when language changes
  useEffect(() => {
    if (isTranslated) {
      // If already translated and language changes, re-translate
      const translateAgain = async () => {
        restoreOriginalContent();
        setIsTranslated(false);

        // Wait a moment for the content to reset, then translate again
        setTimeout(async () => {
          const content = extractPageContent();
          setOriginalContent(content);

          const translations = await translationService.translateBatch(content, targetLanguage);
          applyTranslations(translations);
          setIsTranslated(true);
        }, 100);
      };

      translateAgain();
    }
  }, [targetLanguage]);

  // Clean up when component unmounts
  useEffect(() => {
    return () => {
      if (isTranslated) {
        restoreOriginalContent();
        setIsTranslated(false);
      }
    };
  }, []);

  return (
    <div className="page-translation" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '10px',
      padding: '10px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      margin: '10px 0',
      border: '1px solid #dee2e6'
    }}>
      <select
        value={targetLanguage}
        onChange={(e) => setTargetLanguage(e.target.value)}
        style={{
          padding: '6px 10px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          fontSize: '14px'
        }}
        disabled={isTranslating}
      >
        {languageOptions.map(lang => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </select>

      <button
        onClick={handleTranslatePage}
        disabled={isTranslating}
        style={{
          padding: '8px 16px',
          backgroundColor: isTranslated ? '#28a745' : '#007cba',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: isTranslating ? 'not-allowed' : 'pointer',
          fontSize: '14px',
          fontWeight: '500'
        }}
      >
        {isTranslating ? 'Translating...' : (isTranslated ? 'Restore Original' : 'Translate Page')}
      </button>

      {isTranslated && (
        <span style={{
          fontSize: '12px',
          color: '#28a745',
          fontWeight: '500'
        }}>
          Page translated to {languageOptions.find(l => l.code === targetLanguage)?.name}
        </span>
      )}
    </div>
  );
};

export default PageTranslation;