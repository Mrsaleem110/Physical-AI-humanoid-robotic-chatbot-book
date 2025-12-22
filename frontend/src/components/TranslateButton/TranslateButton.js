import React, { useState, useEffect } from 'react';
import { translateToUrdu } from './TranslationService';
import './TranslateButton.css';

export default function TranslateButton() {
  const [isTranslated, setIsTranslated] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [originalContent, setOriginalContent] = useState(null);
  const [translatedContent, setTranslatedContent] = useState(null);
  const [error, setError] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    // Check if user is logged in by checking AuthContext or sessionStorage
    const user = sessionStorage.getItem('user') || localStorage.getItem('user');
    setIsLoggedIn(!!user);

    // Check if user has preference for Urdu
    const preferUrdu = localStorage.getItem('preferUrdu') === 'true';
    if (preferUrdu && isLoggedIn) {
      setIsTranslated(true);
      // Load cached translation if available
      const cached = sessionStorage.getItem(`translation_${window.location.pathname}`);
      if (cached) {
        setTranslatedContent(cached);
        applyTranslation(cached);
      }
    }
  }, []);

  const getMainContent = () => {
    const contentElement = document.querySelector('.theme-doc-markdown, article.markdown, .markdown');
    return contentElement;
  };

  const extractTextContent = (element) => {
    // Clone the element to avoid modifying the original
    const clone = element.cloneNode(true);
    
    // Remove code blocks and preserve them
    const codeBlocks = clone.querySelectorAll('pre, code');
    codeBlocks.forEach((block, index) => {
      block.setAttribute('data-preserve', `CODE_BLOCK_${index}`);
    });
    
    return clone.innerHTML;
  };

  const applyTranslation = (translatedHtml) => {
    const contentElement = getMainContent();
    if (contentElement) {
      contentElement.innerHTML = translatedHtml;
      contentElement.classList.add('urdu-mode');
      
      // Dispatch event for Root.js to handle styling
      window.dispatchEvent(new Event('urduTranslationEnabled'));
    }
  };

  const restoreOriginal = () => {
    const contentElement = getMainContent();
    if (contentElement && originalContent) {
      contentElement.innerHTML = originalContent;
      contentElement.classList.remove('urdu-mode');
      
      // Dispatch event for Root.js
      window.dispatchEvent(new Event('urduTranslationDisabled'));
    }
  };

  const handleTranslate = async () => {
    if (isTranslated) {
      // Switch back to original
      setIsTranslated(false);
      restoreOriginal();
      localStorage.setItem('preferUrdu', 'false');
      return;
    }

    // Check if we already have a translation
    if (translatedContent) {
      setIsTranslated(true);
      applyTranslation(translatedContent);
      localStorage.setItem('preferUrdu', 'true');
      return;
    }

    // Perform new translation
    setIsTranslating(true);
    setError(null);

    try {
      const contentElement = getMainContent();
      if (!contentElement) {
        throw new Error('Content not found');
      }

      // Store original content
      const original = contentElement.innerHTML;
      setOriginalContent(original);

      // Extract text for translation
      const contentToTranslate = extractTextContent(contentElement);

      // Call translation service
      const translated = await translateToUrdu(contentToTranslate);

      // Store translation
      setTranslatedContent(translated);
      sessionStorage.setItem(`translation_${window.location.pathname}`, translated);

      // Apply translation
      setIsTranslated(true);
      applyTranslation(translated);
      localStorage.setItem('preferUrdu', 'true');

    } catch (err) {
      console.error('Translation error:', err);
      console.error('Error stack:', err.stack);
      setError(err.message || 'Translation failed. Please try again.');
      
      // Also show error in alert for debugging
      alert(`Translation Error: ${err.message}\n\nCheck browser console for details.`);
    } finally {
      setIsTranslating(false);
    }
  };

  // Always show button (removed user check)
  return (
    <div className="translate-button-container">
      <button
        onClick={handleTranslate}
        disabled={isTranslating}
        className={`translate-button ${isTranslated ? 'active' : ''}`}
        title={isTranslated ? 'Show Original (English)' : 'Translate to Urdu'}
      >
        <span className="translate-icon">
          {isTranslating ? '⏳' : isTranslated ? '🇬🇧' : '🇵🇰'}
        </span>
        <span className="translate-text">
          {isTranslating ? 'Translating...' : isTranslated ? 'Show Original' : 'Translate to Urdu'}
        </span>
      </button>
      {error && (
        <div className="translate-error">
          {error}
        </div>
      )}
    </div>
  );
}

  const getMainContent = () => {
    const contentElement = document.querySelector('.theme-doc-markdown, article.markdown, .markdown');
    return contentElement;
  };

  const extractTextContent = (element) => {
    // Clone the element to avoid modifying the original
    const clone = element.cloneNode(true);
    
    // Remove code blocks and preserve them
    const codeBlocks = clone.querySelectorAll('pre, code');
    codeBlocks.forEach((block, index) => {
      block.setAttribute('data-preserve', `CODE_BLOCK_${index}`);
    });
    
    return clone.innerHTML;
  };

  const applyTranslation = (translatedHtml) => {
    const contentElement = getMainContent();
    if (contentElement) {
      contentElement.innerHTML = translatedHtml;
      contentElement.classList.add('urdu-mode');
      
      // Dispatch event for Root.js to handle styling
      window.dispatchEvent(new Event('urduTranslationEnabled'));
    }
  };

  const restoreOriginal = () => {
    const contentElement = getMainContent();
    if (contentElement && originalContent) {
      contentElement.innerHTML = originalContent;
      contentElement.classList.remove('urdu-mode');
      
      // Dispatch event for Root.js
      window.dispatchEvent(new Event('urduTranslationDisabled'));
    }
  };

  const handleTranslate = async () => {
    if (isTranslated) {
      // Switch back to original
      setIsTranslated(false);
      restoreOriginal();
      localStorage.setItem('preferUrdu', 'false');
      return;
    }

    // Check if we already have a translation
    if (translatedContent) {
      setIsTranslated(true);
      applyTranslation(translatedContent);
      localStorage.setItem('preferUrdu', 'true');
      return;
    }

    // Perform new translation
    setIsTranslating(true);
    setError(null);

    try {
      const contentElement = getMainContent();
      if (!contentElement) {
        throw new Error('Content not found');
      }

      // Store original content
      const original = contentElement.innerHTML;
      setOriginalContent(original);

      // Extract text for translation
      const contentToTranslate = extractTextContent(contentElement);

      // Call translation service
      const translated = await translateToUrdu(contentToTranslate);

      // Store translation
      setTranslatedContent(translated);
      sessionStorage.setItem(`translation_${window.location.pathname}`, translated);

      // Apply translation
      setIsTranslated(true);
      applyTranslation(translated);
      localStorage.setItem('preferUrdu', 'true');

    } catch (err) {
      console.error('Translation error:', err);
      console.error('Error stack:', err.stack);
      setError(err.message || 'Translation failed. Please try again.');
      
      // Also show error in alert for debugging
      alert(`Translation Error: ${err.message}\n\nCheck browser console for details.`);
    } finally {
      setIsTranslating(false);
    }
  };

  // Always show button (removed user check)
  return (
    <div className="translate-button-container">
      <button
        onClick={handleTranslate}
        disabled={isTranslating}
        className={`translate-button ${isTranslated ? 'active' : ''}`}
        title={isTranslated ? 'Show Original (English)' : 'Translate to Urdu'}
      >
        <span className="translate-icon">
          {isTranslating ? '⏳' : isTranslated ? '🇬🇧' : '🇵🇰'}
        </span>
        <span className="translate-text">
          {isTranslating ? 'Translating...' : isTranslated ? 'Show Original' : 'Translate to Urdu'}
        </span>
      </button>
      {error && (
        <div className="translate-error">
          {error}
        </div>
      )}
    </div>
  );
}
