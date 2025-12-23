// Controlled translation functionality that gives users more control over language switching
// Provides preference controls and keyboard shortcuts

// Set up translation configuration
window.translationConfig = {
  apiUrl: window.env?.REACT_APP_LIBRETRANSLATE_URL || 'http://localhost:5000/translate'
};

// Local storage key used by the multilingual dropdown
const PREF_KEY = 'preferred_language';

// Safe helper to read stored preference
function getStoredPreference() {
  try { return localStorage.getItem(PREF_KEY); } catch (e) { return null; }
}

// Safe helper to write stored preference
function setStoredPreference(lang) {
  try { localStorage.setItem(PREF_KEY, lang); } catch (e) { }
}

// Apply document attributes for accessibility/direction
function applyLangAttributes(lang) {
  if (!lang) return;
  document.documentElement.lang = lang;
  const rtlLanguages = ['ur', 'ar', 'fa', 'he', 'ps', 'sd', 'ug', 'ku', 'yi'];
  document.documentElement.dir = rtlLanguages.includes(lang.toLowerCase()) ? 'rtl' : 'ltr';
}

// When other components dispatch a 'translate' event, persist and apply attributes
window.addEventListener('translate', function(e) {
  const lang = e?.detail?.lang;
  if (!lang) return;
  setStoredPreference(lang);
  applyLangAttributes(lang);
});

// On load, if a stored preference exists and current URL is not localized to it,
// navigate to the language-prefixed URL so Docusaurus serves localized pages.
document.addEventListener('DOMContentLoaded', function() {
  const stored = getStoredPreference();
  if (!stored) return;
  const pathParts = window.location.pathname.split('/');
  const currentLocaleInPath = pathParts[1];
  if (currentLocaleInPath !== stored) {
    const currentHost = window.location.origin;
    let targetPath;
    if (currentLocaleInPath && currentLocaleInPath.length === 2) {
      targetPath = '/' + pathParts.slice(2).join('/');
    } else {
      targetPath = window.location.pathname;
    }
    // Ensure trailing slash for root
    if (targetPath === '/' || targetPath === '') targetPath = '/';
    window.location.href = currentHost + '/' + stored + targetPath;
  } else {
    applyLangAttributes(currentLocaleInPath);
  }
});

// Expose helper API for other scripts
window.controlledTranslation = window.controlledTranslation || {
  getPreferred: getStoredPreference,
  setPreferred: function(l) { setStoredPreference(l); applyLangAttributes(l); }
};


