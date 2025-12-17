import React, { useState, useEffect } from 'react';
import translationService from '../services/translationService';

const TranslationPanel = () => {
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLanguage, setSourceLanguage] = useState('en');
  const [targetLanguage, setTargetLanguage] = useState('ur');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [supportedLanguages] = useState([
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'ur', name: 'Urdu', flag: '🇵🇰' },
    { code: 'hi', name: 'Hindi', flag: '🇮🇳' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵' },
    { code: 'ko', name: 'Korean', flag: '🇰🇷' },
    { code: 'ar', name: 'Arabic', flag: '🇸🇦' }
  ]);

  const handleTranslate = async () => {
    if (!sourceText.trim()) return;

    setLoading(true);
    setError(null);
    setTranslatedText('');

    try {
      const translated = await translationService.translateText(sourceText, targetLanguage, sourceLanguage);
      setTranslatedText(translated);

      // Add to history
      const newTranslation = {
        id: Date.now(),
        original: sourceText,
        translated: translated,
        sourceLang: sourceLanguage,
        targetLang: targetLanguage,
        sourceLangName: supportedLanguages.find(l => l.code === sourceLanguage)?.name || sourceLanguage,
        targetLangName: supportedLanguages.find(l => l.code === targetLanguage)?.name || targetLanguage,
        timestamp: new Date().toISOString()
      };
      setHistory(prev => [newTranslation, ...prev.slice(0, 9)]); // Keep last 10 translations
    } catch (error) {
      console.error('Translation error:', error);
      setError('Translation failed. Please check your API configuration.');
      setTranslatedText('Error occurred during translation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const swapLanguages = () => {
    const tempLang = sourceLanguage;
    setSourceLanguage(targetLanguage);
    setTargetLanguage(tempLang);

    const tempText = sourceText;
    setSourceText(translatedText);
    setTranslatedText(tempText);
  };

  const clearText = () => {
    setSourceText('');
    setTranslatedText('');
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  // CSS styles for the translation panel
  const styles = {
    panel: {
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '16px',
      margin: '1rem 0',
      backgroundColor: 'white',
      fontFamily: 'system-ui, sans-serif',
      maxWidth: '800px',
    },
    title: {
      fontSize: '1.2rem',
      fontWeight: 'bold',
      marginBottom: '16px',
      color: '#282c34',
    },
    controls: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: '16px',
      gap: '8px',
    },
    languageSelector: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
    },
    languageLabel: {
      fontSize: '0.8rem',
      marginBottom: '4px',
      color: '#555',
    },
    select: {
      padding: '4px 8px',
      border: '1px solid #ccc',
      borderRadius: '4px',
      fontSize: '0.9rem',
    },
    swapButton: {
      padding: '8px 12px',
      margin: '0 8px',
      backgroundColor: '#007cba',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '1rem',
    },
    translationArea: {
      display: 'flex',
      gap: '16px',
      marginBottom: '16px',
    },
    textContainer: {
      flex: 1,
      display: 'flex',
      flexDirection: 'column',
    },
    textArea: {
      width: '100%',
      padding: '8px',
      border: '1px solid #ccc',
      borderRadius: '4px',
      resize: 'vertical',
      fontSize: '0.9rem',
      fontFamily: 'inherit',
    },
    textActions: {
      display: 'flex',
      justifyContent: 'flex-end',
      gap: '4px',
      marginTop: '4px',
    },
    actionButton: {
      padding: '4px 8px',
      border: '1px solid #ccc',
      backgroundColor: '#f5f5f5',
      cursor: 'pointer',
      borderRadius: '4px',
      fontSize: '0.8rem',
    },
    translationButtons: {
      display: 'flex',
      justifyContent: 'center',
      gap: '12px',
      marginBottom: '16px',
    },
    translateButton: {
      padding: '8px 16px',
      backgroundColor: '#007cba',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.9rem',
    },
    clearButton: {
      padding: '8px 16px',
      backgroundColor: '#6c757d',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.9rem',
    },
    translationResult: {
      marginTop: '16px',
      padding: '12px',
      backgroundColor: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: '4px',
    },
    resultTitle: {
      fontSize: '1rem',
      fontWeight: 'bold',
      marginBottom: '8px',
      color: '#282c34',
    },
    translationHistory: {
      marginTop: '16px',
    },
    historyTitle: {
      fontSize: '1rem',
      fontWeight: 'bold',
      marginBottom: '8px',
      color: '#282c34',
    },
    historyList: {
      maxHeight: '200px',
      overflowY: 'auto',
    },
    historyItem: {
      padding: '8px',
      borderBottom: '1px solid #eee',
    },
    historyOriginal: {
      fontWeight: 'bold',
      marginBottom: '4px',
    },
    historyTranslated: {
      marginBottom: '4px',
    },
    historyActions: {
      textAlign: 'right',
    },
    historyActionBtn: {
      padding: '2px 6px',
      border: '1px solid #ccc',
      backgroundColor: '#f5f5f5',
      cursor: 'pointer',
      borderRadius: '4px',
      fontSize: '0.8rem',
    },
  };

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>Text Translation</h3>

      <div style={styles.controls}>
        <div style={styles.languageSelector}>
          <label style={styles.languageLabel} htmlFor="source-lang">From:</label>
          <select
            id="source-lang"
            value={sourceLanguage}
            onChange={(e) => setSourceLanguage(e.target.value)}
            style={styles.select}
          >
            {supportedLanguages.map(lang => (
              <option key={lang.code} value={lang.code}>
                {lang.flag} {lang.name}
              </option>
            ))}
          </select>
        </div>

        <button style={styles.swapButton} onClick={swapLanguages}>
          ↔
        </button>

        <div style={styles.languageSelector}>
          <label style={styles.languageLabel} htmlFor="target-lang">To:</label>
          <select
            id="target-lang"
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            style={styles.select}
          >
            {supportedLanguages.map(lang => (
              <option key={lang.code} value={lang.code}>
                {lang.flag} {lang.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div style={styles.translationArea}>
        <div style={styles.textContainer}>
          <textarea
            value={sourceText}
            onChange={(e) => setSourceText(e.target.value)}
            placeholder={`Enter text to translate from ${supportedLanguages.find(l => l.code === sourceLanguage)?.name}`}
            rows={6}
            style={styles.textArea}
          />
          <div style={styles.textActions}>
            <button onClick={() => setSourceText('')} style={styles.actionButton}>
              Clear
            </button>
            <button onClick={() => copyToClipboard(sourceText)} style={styles.actionButton}>
              Copy
            </button>
          </div>
        </div>

        <div style={styles.textContainer}>
          <textarea
            value={translatedText}
            readOnly
            placeholder={`Translated text in ${supportedLanguages.find(l => l.code === targetLanguage)?.name}`}
            rows={6}
            style={styles.textArea}
          />
          <div style={styles.textActions}>
            <button onClick={() => copyToClipboard(translatedText)} style={styles.actionButton}>
              Copy
            </button>
          </div>
        </div>
      </div>

      <div style={styles.translationButtons}>
        <button
          onClick={handleTranslate}
          disabled={loading || !sourceText.trim()}
          style={styles.translateButton}
        >
          {loading ? 'Translating...' : 'Translate'}
        </button>
        <button onClick={clearText} style={styles.clearButton}>
          Clear All
        </button>
      </div>

      {error && (
        <div style={{
          ...styles.translationResult,
          backgroundColor: '#f8d7da',
          border: '1px solid #f5c6cb',
          color: '#721c24'
        }}>
          <h4 style={styles.resultTitle}>Translation Error</h4>
          <p>{error}</p>
        </div>
      )}

      {translatedText && !error && (
        <div style={styles.translationResult}>
          <h4 style={styles.resultTitle}>Translation Result</h4>
          <p>{translatedText}</p>
        </div>
      )}

      {history.length > 0 && (
        <div style={styles.translationHistory}>
          <h4 style={styles.historyTitle}>Recent Translations</h4>
          <div style={styles.historyList}>
            {history.map(item => (
              <div key={item.id} style={styles.historyItem}>
                <div style={styles.historyOriginal}>
                  <strong>{supportedLanguages.find(l => l.code === item.sourceLang)?.flag} {supportedLanguages.find(l => l.code === item.sourceLang)?.name}:</strong> {item.original.substring(0, 50)}{item.original.length > 50 ? '...' : ''}
                </div>
                <div style={styles.historyTranslated}>
                  <strong>{supportedLanguages.find(l => l.code === item.targetLang)?.flag} {supportedLanguages.find(l => l.code === item.targetLang)?.name}:</strong> {item.translated.substring(0, 50)}{item.translated.length > 50 ? '...' : ''}
                </div>
                <div style={styles.historyActions}>
                  <button onClick={() => copyToClipboard(item.translated)} style={styles.historyActionBtn}>Copy</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TranslationPanel;