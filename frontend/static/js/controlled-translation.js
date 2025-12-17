// Controlled translation functionality that gives users more control over language switching
// Provides preference controls and keyboard shortcuts

// Set up translation configuration
window.translationConfig = {
  apiUrl: window.env?.REACT_APP_LIBRETRANSLATE_URL || 'http://localhost:5000/translate'
};


