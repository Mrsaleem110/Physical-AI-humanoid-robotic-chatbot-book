import React, { useState, useEffect } from 'react';
import { useUser } from '../contexts/UserContext';
import translationService from '../services/translationService';

const ChapterTranslation = ({ chapterId, content }) => {
  const { user } = useUser();
  const [isTranslated, setIsTranslated] = useState(false);
  const [targetLanguage, setTargetLanguage] = useState('ur');
  const [loading, setLoading] = useState(false);
  const [translatedContent, setTranslatedContent] = useState('');
  const [error, setError] = useState(null);

  const languageOptions = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'ur', name: 'Urdu', flag: '🇵🇰' }
  ];

  const handleTranslate = async () => {
    if (!user) {
      alert('Please log in to use translation feature');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (isTranslated) {
        // Remove translation
        setIsTranslated(false);
      } else {
        // Perform translation
        const translated = await translationService.translateText(content, targetLanguage, 'en');
        setTranslatedContent(translated);
        setIsTranslated(true);
      }
    } catch (err) {
      console.error('Translation error:', err);
      setError('Translation failed. Please try again later.');
      alert('Translation failed. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = (e) => {
    setTargetLanguage(e.target.value);
    // Reset translation state when language changes
    if (isTranslated) {
      setIsTranslated(false);
    }
  };

  // Reset when content changes
  useEffect(() => {
    setIsTranslated(false);
    setTranslatedContent('');
    setError(null);
  }, [content]);

  const currentLanguageName = languageOptions.find(lang => lang.code === targetLanguage)?.name || 'Unknown';
  const currentLanguageFlag = languageOptions.find(lang => lang.code === targetLanguage)?.flag || '';

  return (
    <div className="chapter-translation" style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '12px',
      padding: '20px',
      margin: '20px 0',
      boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h4 style={{
          margin: '0',
          color: 'white',
          fontSize: '18px',
          fontWeight: '600'
        }}>
          🌐 Content Translation
        </h4>
        <button
          onClick={handleTranslate}
          disabled={loading}
          style={{
            padding: '10px 16px',
            backgroundColor: loading ? '#6c757d' : (isTranslated ? '#28a745' : 'rgba(255,255,255,0.2)'),
            color: 'white',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: '600',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}
        >
          {loading ? 'Translating...' : (isTranslated ? 'Remove Translation' : 'Translate')}
        </button>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label style={{
          display: 'block',
          marginBottom: '6px',
          color: 'rgba(255,255,255,0.9)',
          fontWeight: '500',
          fontSize: '14px'
        }}>
          📋 Select Language:
        </label>
        <select
          value={targetLanguage}
          onChange={handleLanguageChange}
          style={{
            width: '100%',
            padding: '10px 12px',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            backgroundColor: 'rgba(255,255,255,0.95)',
            fontSize: '14px',
            color: '#333',
            transition: 'all 0.3s ease'
          }}
          disabled={loading}
        >
          {languageOptions.map(lang => (
            <option key={lang.code} value={lang.code}>
              {lang.flag} {lang.name}
            </option>
          ))}
        </select>
      </div>

      {error && (
        <div style={{
          marginBottom: '15px',
          padding: '10px',
          backgroundColor: 'rgba(220, 53, 69, 0.2)',
          border: '1px solid rgba(220, 53, 69, 0.3)',
          borderRadius: '6px',
          color: 'white'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <div style={{
        marginTop: '15px',
        padding: '12px',
        backgroundColor: 'rgba(255,255,255,0.1)',
        borderRadius: '8px',
        backdropFilter: 'blur(10px)'
      }}>
        <p style={{
          margin: '0 0 8px 0',
          color: 'rgba(255,255,255,0.9)',
          fontSize: '14px'
        }}>
          <strong style={{ color: 'white' }}>Status:</strong> {isTranslated
            ? `🌐 Content translated to ${currentLanguageFlag} ${currentLanguageName}`
            : '🔤 Using original language'}
        </p>
        <p style={{
          margin: '0',
          color: 'rgba(255,255,255,0.9)',
          fontSize: '14px'
        }}>
          <strong style={{ color: 'white' }}>Supported Languages:</strong> {languageOptions.map(lang => lang.name).join(', ')}
        </p>
      </div>

      {/* Render translated content if available */}
      {isTranslated && !loading && (
        <div style={{
          marginTop: '15px',
          padding: '15px',
          backgroundColor: 'white',
          borderRadius: '8px',
          border: '1px solid #ddd',
          minHeight: '50px'
        }}>
          <h5 style={{ margin: '0 0 10px 0', color: '#333', fontSize: '16px' }}>
            Translated Content ({currentLanguageFlag} {currentLanguageName}):
          </h5>
          <div style={{ color: '#444', lineHeight: '1.6' }}>
            {translatedContent}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChapterTranslation;