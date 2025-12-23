// Translation configuration for Docusaurus
// This file sets up environment variables for the translation service

// Set up global configuration for translation service
window.translationConfig = window.translationConfig || {
  apiUrl: 'http://localhost:5000/translate'
};

// Function to update the API URL if needed
window.updateTranslationConfig = function(config) {
  window.translationConfig = {
    ...window.translationConfig,
    ...config
  };
};