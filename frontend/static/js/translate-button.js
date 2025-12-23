// Mount TranslateButton to navbar after page loads
(function() {
  console.log('[TranslateButton] Script loaded');

  let attempts = 0;
  const maxAttempts = 20;
  let button = null;
  let isTranslated = false;
  let originalContent = null;
  let translatedContent = null;
  let isTranslating = false;
  let currentPath = window.location.pathname; // Track current path for content caching

  // Function to convert Urdu script to Roman Urdu (transliteration)
  function urduToRoman(urduText) {
    if (!urduText) return urduText;

    // More comprehensive mapping of Urdu characters to Roman characters
    const urduToRomanMap = {
      // Alphabets
      'ا': 'a', 'آ': 'aa', 'أ': 'a', 'إ': 'i', 'ئ': 'e', 'ؤ': 'o', 'ب': 'b', 'پ': 'p', 'ت': 't',
      'ٹ': 't', 'ث': 's', 'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd',
      'ذ': 'z', 'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's',
      'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q', 'ک': 'k',
      'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'و': 'w', 'ؤ': 'w', 'ه': 'h', 'ھ': 'h',
      'ء': 'e', 'ۃ': 't', 'ی': 'y', 'ے': 'e', 'ي': 'y', 'ى': 'y', 'ۓ': 'ye',

      // Vowel signs/diacritics
      'َ': 'a', 'ُ': 'u', 'ِ': 'i', 'ّ': '', 'ً': 'an', 'ٌ': 'un', 'ٍ': 'in', 'ْ': '', 'ٰ': 'aa', 'ٔ': 'e',

      // Numbers
      '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
      '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',

      // Punctuation
      '،': ',', '۔': '.', '؟': '?', '؛': ';', '٪': '%', '٫': '.', '٬': ',', '﴿': '"', '﴾': '"',
      '«': '"', '»': '"', '(': '(', ')': ')', '[': '[', ']': ']'
    };

    let romanText = urduText;

    // Create a sorted array of keys by length (descending) to handle multi-character sequences first
    const sortedKeys = Object.keys(urduToRomanMap).sort((a, b) => b.length - a.length);

    // Replace each Urdu character with its Roman equivalent
    for (const urduChar of sortedKeys) {
      const romanChar = urduToRomanMap[urduChar];
      // Use global replacement for all instances
      const regex = new RegExp(urduChar.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
      romanText = romanText.replace(regex, romanChar);
    }

    // Handle common combined characters and ligatures
    romanText = romanText
      // Common combinations
      .replace(/ک/g, 'k')
      .replace(/گ/g, 'g')
      .replace(/چ/g, 'ch')
      .replace(/ج/g, 'j')
      .replace(/ض/g, 'z')
      .replace(/ظ/g, 'z')
      .replace(/ط/g, 't')
      .replace(/ذ/g, 'z')
      .replace(/ث/g, 's')
      .replace(/ص/g, 's')
      .replace(/ش/g, 'sh')
      .replace(/ژ/g, 'zh')
      .replace(/ح/g, 'h')
      .replace(/خ/g, 'kh')
      .replace(/ғ/g, 'gh')
      .replace(/ڤ/g, 'p')
      .replace(/چ/g, 'ch')
      .replace(/ڙ/g, 'r')
      .replace(/ڍ/g, 'd')
      .replace(/ڀ/g, 'bh')
      .replace(/ﷲ/g, 'Allah')
      .replace(/ Muhammad /g, ' Muhammad ')
      .replace(/ Peace /g, ' Peace ')
      // Clean up multiple spaces
      .replace(/\s+/g, ' ')
      // Remove extra spaces around punctuation
      .replace(/\s+([,.!?;:])/, '$1');

    return romanText;
  }

  // Alternative approach: Use a transliteration service if available
  async function transliterateToRoman(urduText) {
    // Fallback to manual transliteration immediately since API approach doesn't work well for general text
    return urduToRoman(urduText);
  }

  // Global API function with better error handling - now with Roman Urdu support
  async function fetchTranslation(text) {
    try {
      if (!text || text.trim().length === 0) {
        return text;
      }

      // Handle long text by splitting into sentences
      if (text.length > 500) {
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        const translations = await Promise.all(
          sentences.map(s => fetchTranslation(s.trim()))
        );
        return translations.join(' ');
      }

      // First, translate to Urdu script
      let urduTranslation = text; // Default to original text

      // Try multiple translation APIs for Urdu script
      const apis = [
        `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=en|ur`,
        `https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ur&dt=t&q=${encodeURIComponent(text)}`
      ];

      for (const apiUrl of apis) {
        try {
          const response = await fetch(apiUrl);
          if (!response.ok) continue;

          const data = await response.json();

          if (data && data.responseData && data.responseData.translatedText) {
            const translated = data.responseData.translatedText;
            if (translated && translated !== 'REQUEST LANGUAGE PAIR NOT SUPPORTED' && translated.trim() && translated !== text) {
              urduTranslation = translated;
              break;
            }
          } else if (data && Array.isArray(data) && data[0] && Array.isArray(data[0])) {
            // Google Translate API format
            const translated = data[0].map(item => item[0]).join('') || text;
            if (translated && translated !== text && translated.trim()) {
              urduTranslation = translated;
              break;
            }
          }
        } catch (apiError) {
          console.warn('[TranslateButton] Translation API failed:', apiUrl, apiError);
          continue;
        }
      }

      // Check user preference for script type
      const scriptPreference = localStorage.getItem('urduScriptPreference') || 'urdu'; // Default to Urdu script

      if (scriptPreference === 'roman') {
        // Convert the Urdu script to Roman Urdu
        const romanUrdu = await transliterateToRoman(urduTranslation);
        console.log('[TranslateButton] Translation to Roman Urdu successful:', text.substring(0, 30) + '... → ' + romanUrdu.substring(0, 30) + '...');
        return romanUrdu;
      } else {
        // Return Urdu script as is
        console.log('[TranslateButton] Translation to Urdu script successful:', text.substring(0, 30) + '... → ' + urduTranslation.substring(0, 30) + '...');
        return urduTranslation;
      }

    } catch (e) {
      console.error('[TranslateButton] Translation error:', e);
      return text;
    }
  }

  // Function to update content tracking when page changes
  function updateContentTracking() {
    const newPath = window.location.pathname;
    if (newPath !== currentPath) {
      console.log('[TranslateButton] Path changed from', currentPath, 'to', newPath);
      currentPath = newPath;

      // If we were translated, we need to get fresh original content for the new page
      if (isTranslated) {
        const contentElement = getMainContent();
        if (contentElement) {
          // Save current translated content to session storage for this path
          sessionStorage.setItem('translation_' + newPath, contentElement.innerHTML);
          // Get new original content
          const newOriginalContent = contentElement.innerHTML;
          // Restore to original content first, then translate again if needed
          contentElement.innerHTML = newOriginalContent;
          contentElement.classList.remove('urdu-mode');
          removeUrduStyles();

          // Then translate the new content if user prefers Urdu
          setTimeout(() => {
            if (localStorage.getItem('preferUrdu') === 'true') {
              handleTranslate();
            }
          }, 100);
        }
      } else {
        // If not translated, just update the original content for the new path
        const contentElement = getMainContent();
        if (contentElement) {
          originalContent = contentElement.innerHTML;
        }
      }
    }
  }

  // Translate HTML content
  async function translateContent(htmlContent) {
    console.log('[TranslateButton] Starting HTML translation');
    
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlContent, 'text/html');
    
    // Get all text nodes
    const walker = document.createTreeWalker(
      doc.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: (node) => {
          const parent = node.parentElement;
          if (!node.textContent.trim()) return NodeFilter.FILTER_REJECT;
          if (parent && (parent.tagName === 'CODE' || parent.tagName === 'PRE')) return NodeFilter.FILTER_REJECT;
          if (parent && parent.tagName === 'SCRIPT') return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );

    const segments = [];
    let node;
    while (node = walker.nextNode()) {
      const text = node.textContent.trim();
      if (text.length > 0 && text.length < 1000) {
        segments.push({ node, original: text });
      }
    }

    console.log('[TranslateButton] Found ' + segments.length + ' text segments');

    if (segments.length === 0) {
      console.warn('[TranslateButton] No segments found');
      return htmlContent;
    }

    // Translate in batches
    const batchSize = 3;
    for (let i = 0; i < segments.length; i += batchSize) {
      const batch = segments.slice(i, i + batchSize);
      console.log('[TranslateButton] Translating batch ' + (i / batchSize + 1) + '/' + Math.ceil(segments.length / batchSize));
      
      // Translate each segment in parallel
      const promises = batch.map(async (segment) => {
        try {
          const translated = await fetchTranslation(segment.original);
          if (translated && translated !== segment.original) {
            segment.node.textContent = translated;
            console.log('[TranslateButton] Translated: "' + segment.original.substring(0, 30) + '..." → "' + translated.substring(0, 30) + '..."');
          }
        } catch (e) {
          console.warn('[TranslateButton] Failed to translate segment:', e);
        }
      });

      await Promise.all(promises);
      
      // Delay between batches
      if (i + batchSize < segments.length) {
        await new Promise(r => setTimeout(r, 500));
      }
    }

    const result = doc.body.innerHTML;
    console.log('[TranslateButton] HTML translation complete');
    return result;
  }

  function getMainContent() {
    // Try different selectors
    const selectors = [
      '.theme-doc-markdown',
      'article.markdown',
      'article',
      'main',
      '[role="main"]',
      '.markdown'
    ];

    for (const selector of selectors) {
      const el = document.querySelector(selector);
      if (el && el.textContent.trim()) {
        console.log('[TranslateButton] Found content with selector: ' + selector);
        return el;
      }
    }

    console.warn('[TranslateButton] Could not find content element with any selector');
    return null;
  }

  function ensureButtonVisible() {
    if (button) {
      // Use CSS class to ensure visibility instead of inline styles
      button.classList.add('translate-button-visible');
      button.style.display = 'flex';
      button.style.opacity = '1';
      button.style.visibility = 'visible';
      button.style.pointerEvents = 'auto';
      button.style.zIndex = '9999';
      console.log('[TranslateButton] Button visibility ensured');
    }

    // Also ensure the container is visible
    const container = document.querySelector('.translate-button-container');
    if (container) {
      container.style.display = 'flex';
      container.style.opacity = '1';
      container.style.visibility = 'visible';
      container.style.zIndex = '9999';
    }
  }

  function attachClickHandler() {
    if (!button) {
      console.warn('[TranslateButton] Button not found, cannot attach handler');
      return;
    }
    
    console.log('[TranslateButton] Attaching click handler');
    button.onclick = null;
    button.removeEventListener('click', handleButtonClick);
    button.addEventListener('click', handleButtonClick);
  }

  function handleButtonClick(e) {
    e.preventDefault();
    e.stopPropagation();
    console.log('[TranslateButton] Button clicked, isTranslating=' + isTranslating + ', isTranslated=' + isTranslated);

    if (isTranslating) {
      console.log('[TranslateButton] Already translating, ignoring click');
      return;
    }

    if (isTranslated) {
      // If already translated, allow toggling between script types
      const scriptPreference = localStorage.getItem('urduScriptPreference') || 'urdu';
      const newScriptPreference = scriptPreference === 'urdu' ? 'roman' : 'urdu';
      localStorage.setItem('urduScriptPreference', newScriptPreference);

      // Re-translate with new preference
      const contentElement = getMainContent();
      if (contentElement && originalContent) {
        console.log('[TranslateButton] Re-translating with new script preference:', newScriptPreference);
        handleTranslate();
      } else {
        restoreOriginal();
      }
    } else {
      // First time translation - default to Urdu script but allow user to switch
      localStorage.setItem('urduScriptPreference', 'urdu');
      handleTranslate();
    }
  }

  async function handleTranslate() {
    console.log('[TranslateButton] handleTranslate started');
    isTranslating = true;
    button.disabled = true;
    button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">⏳</span><span class="translate-text">Translating...</span>';

    try {
      const contentElement = getMainContent();
      if (!contentElement) {
        throw new Error('Content element not found on page');
      }

      console.log('[TranslateButton] Found content element, starting translation');
      // Save the original content for this specific path
      originalContent = contentElement.innerHTML;
      const originalContentKey = 'original_' + window.location.pathname;
      sessionStorage.setItem(originalContentKey, originalContent);

      // Check if we already have a translated version in cache for this path
      const translationKey = 'translation_' + window.location.pathname;
      let cachedTranslation = sessionStorage.getItem(translationKey);

      if (!cachedTranslation) {
        console.log('[TranslateButton] No cached translation, calling translateContent');
        translatedContent = await translateContent(originalContent);
        console.log('[TranslateButton] Translation completed, caching result');
        sessionStorage.setItem(translationKey, translatedContent);
        cachedTranslation = translatedContent;
      } else {
        console.log('[TranslateButton] Using cached translation');
        translatedContent = cachedTranslation;
      }

      console.log('[TranslateButton] Applying translation to content');
      applyTranslation(cachedTranslation);
      isTranslated = true;
      localStorage.setItem('preferUrdu', 'true');

      // Update button text to reflect script preference
      const scriptPreference = localStorage.getItem('urduScriptPreference') || 'urdu';
      if (scriptPreference === 'roman') {
        button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Switch to Urdu Script</span>';
      } else {
        button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Switch to Roman Urdu</span>';
      }

      button.classList.add('active');
      console.log('[TranslateButton] Translation applied successfully');

      // Ensure button stays visible
      ensureButtonVisible();
      attachClickHandler();

    } catch (error) {
      console.error('[TranslateButton] Error during translation:', error);
      alert('Translation Error: ' + error.message);
      button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Translate to Urdu</span>';
    } finally {
      isTranslating = false;
      button.disabled = false;
      ensureButtonVisible();
      attachClickHandler();
    }
  }

  function applyTranslation(html) {
    const contentElement = getMainContent();
    if (contentElement) {
      console.log('[TranslateButton] Applying translated HTML to content');
      contentElement.innerHTML = html;

      // Check user preference for script type and apply appropriate styles
      const scriptPreference = localStorage.getItem('urduScriptPreference') || 'urdu';
      if (scriptPreference === 'roman') {
        contentElement.classList.remove('urdu-mode');
        contentElement.classList.add('roman-urdu-mode');
      } else {
        contentElement.classList.remove('roman-urdu-mode');
        contentElement.classList.add('urdu-mode');
      }

      addUrduStyles();

      // Ensure button visibility after content update
      setTimeout(() => {
        ensureButtonVisible();
      }, 100);
    }
  }

  function restoreOriginal() {
    const contentElement = getMainContent();
    if (contentElement) {
      console.log('[TranslateButton] Restoring original content');
      // Get the original content for the current path from session storage or use the saved original
      const originalContentKey = 'original_' + window.location.pathname;
      const savedOriginal = sessionStorage.getItem(originalContentKey);

      if (savedOriginal) {
        contentElement.innerHTML = savedOriginal;
        console.log('[TranslateButton] Restored original content from session storage');
      } else if (originalContent) {
        contentElement.innerHTML = originalContent;
        console.log('[TranslateButton] Restored original content from variable');
      } else {
        // If no original content is available, try to get current content before translation
        // This handles cases where original content wasn't saved initially
        const tempOriginal = contentElement.innerHTML;
        contentElement.innerHTML = tempOriginal;
        console.log('[TranslateButton] Restored original content from current DOM');
      }

      contentElement.classList.remove('urdu-mode');
      isTranslated = false;
      localStorage.setItem('preferUrdu', 'false');
      button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Translate to Urdu</span>';
      button.classList.remove('active');
      removeUrduStyles();

      // Ensure button stays visible
      ensureButtonVisible();
      attachClickHandler();
    }
  }

  function addUrduStyles() {
    if (document.getElementById('urdu-styles')) return;
    const style = document.createElement('style');
    style.id = 'urdu-styles';
    style.textContent = `
      .urdu-mode {
        font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif !important;
        direction: rtl !important;
        text-align: right !important;
      }
      .urdu-mode h1, .urdu-mode h2, .urdu-mode h3, .urdu-mode h4, .urdu-mode h5, .urdu-mode h6 {
        direction: rtl !important;
        text-align: right !important;
        line-height: 2 !important;
      }
      .urdu-mode p, .urdu-mode li, .urdu-mode td, .urdu-mode th {
        direction: rtl !important;
        text-align: right !important;
        line-height: 2.2 !important;
      }
      .urdu-mode pre, .urdu-mode code, .urdu-mode .token {
        direction: ltr !important;
        text-align: left !important;
      }
      .urdu-mode ul, .urdu-mode ol {
        padding-right: 2rem;
        padding-left: 0;
      }
      .roman-urdu-mode {
        font-family: 'Noto Sans', 'Arial', sans-serif !important;
        direction: ltr !important;
        text-align: left !important;
      }
      .roman-urdu-mode h1, .roman-urdu-mode h2, .roman-urdu-mode h3, .roman-urdu-mode h4, .roman-urdu-mode h5, .roman-urdu-mode h6 {
        direction: ltr !important;
        text-align: left !important;
        line-height: 1.6 !important;
      }
      .roman-urdu-mode p, .roman-urdu-mode li, .roman-urdu-mode td, .roman-urdu-mode th {
        direction: ltr !important;
        text-align: left !important;
        line-height: 1.6 !important;
      }
      .translate-button.active {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
      }
      .translate-button, .translate-button-visible {
        display: flex !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
        z-index: 9999 !important;
      }
      .translate-button-container {
        display: flex !important;
        opacity: 1 !important;
        visibility: visible !important;
        z-index: 9999 !important;
      }
    `;
    document.head.appendChild(style);
  }

  function removeUrduStyles() {
    const style = document.getElementById('urdu-styles');
    if (style) style.remove();
  }

  function mountTranslateButton() {
    attempts++;

    const mountPoint = document.getElementById('translate-button-mount');

    if (!mountPoint) {
      console.log('[TranslateButton] Mount point not found (attempt ' + attempts + '/' + maxAttempts + ')');
      if (attempts < maxAttempts) {
        setTimeout(mountTranslateButton, 500);
      }
      return;
    }

    if (mountPoint.querySelector('.translate-button-container')) {
      console.log('[TranslateButton] Already mounted');
      return;
    }

    console.log('[TranslateButton] Mounting component');

    const html = `
      <div class="translate-button-container">
        <button
          id="translate-btn"
          class="translate-button"
          title="Translate to Urdu"
          style="display: flex; align-items: center; justify-content: center; gap: 0.75rem; padding: 0.875rem 1.5rem; font-size: 0.95rem; font-weight: 600; color: white; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); border: none; border-radius: 8px; cursor: pointer; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2), 0 1px 2px rgba(37, 99, 235, 0.12); position: relative; z-index: 9999;"
        >
          <span class="translate-icon" style="font-size: 1.25rem; line-height: 1; display: flex; align-items: center; justify-content: center;">🇵🇰</span>
          <span class="translate-text" style="line-height: 1; letter-spacing: 0.01em;">Translate to Urdu</span>
        </button>
      </div>
    `;

    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = html;
    mountPoint.innerHTML = '';
    mountPoint.appendChild(tempDiv.firstElementChild);

    button = document.getElementById('translate-btn');
    if (button) {
      console.log('[TranslateButton] Button found, setting up');
      attachClickHandler();

      // Check for saved preference on initial load
      const preferUrdu = localStorage.getItem('preferUrdu') === 'true';
      if (preferUrdu) {
        const cached = sessionStorage.getItem('translation_' + window.location.pathname);
        if (cached) {
          console.log('[TranslateButton] Restoring cached translation on page load');
          isTranslated = true;
          translatedContent = cached;
          applyTranslation(cached);
          // Update button text to reflect script preference when restoring cached translation
          const scriptPreference = localStorage.getItem('urduScriptPreference') || 'urdu';
          if (scriptPreference === 'roman') {
            button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Switch to Urdu Script</span>';
          } else {
            button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Switch to Roman Urdu</span>';
          }
          button.classList.add('active');
        }
      }

      // Set up mutation observer to watch for changes that might hide the button
      setupMutationObserver();

      console.log('[TranslateButton] Mount complete');
    } else {
      console.error('[TranslateButton] Button element not found after mounting');
    }
  }

  function setupMutationObserver() {
    // Create a mutation observer to watch for changes that might hide the button
    const observer = new MutationObserver(function(mutations) {
      mutations.forEach(function(mutation) {
        // Check if any changes might affect the button visibility
        if (mutation.type === 'attributes' &&
            (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {
          const target = mutation.target;
          if (target === button ||
              target.classList.contains('translate-button') ||
              target.classList.contains('translate-button-container')) {
            // Ensure button visibility after attribute changes
            setTimeout(() => {
              ensureButtonVisible();
            }, 50);
          }
        }

        // Also check if any child changes might affect the button
        if (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0) {
          let shouldCheckVisibility = false;
          for (let node of mutation.addedNodes) {
            if (node === button ||
                (node.nodeType === 1 &&
                 (node.classList.contains('translate-button') ||
                  node.classList.contains('translate-button-container')))) {
              shouldCheckVisibility = true;
              break;
            }
          }
          for (let node of mutation.removedNodes) {
            if (node === button ||
                (node.nodeType === 1 &&
                 (node.classList.contains('translate-button') ||
                  node.classList.contains('translate-button-container')))) {
              shouldCheckVisibility = true;
              break;
            }
          }

          if (shouldCheckVisibility) {
            setTimeout(() => {
              ensureButtonVisible();
            }, 50);
          }
        }
      });
    });

    // Start observing the document body for changes
    observer.observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ['style', 'class']
    });

    console.log('[TranslateButton] Mutation observer set up');
  }

  // Start mounting
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountTranslateButton);
  } else {
    mountTranslateButton();
  }

  // Also try mounting immediately in case DOM is ready
  setTimeout(mountTranslateButton, 100);

  // Periodically check that button remains visible
  setInterval(() => {
    if (isTranslated && button) {
      ensureButtonVisible();
    }
  }, 1000); // Check every second to ensure button stays visible

  // Listen for Docusaurus client-side navigation events
  // This handles navigation between chapters/pages
  let lastPath = window.location.pathname;
  setInterval(() => {
    if (window.location.pathname !== lastPath) {
      console.log('[TranslateButton] Detected page navigation from', lastPath, 'to', window.location.pathname);
      lastPath = window.location.pathname;
      updateContentTracking();

      // Ensure the button is mounted on the new page
      setTimeout(() => {
        if (!document.getElementById('translate-btn')) {
          mountTranslateButton();
        } else {
          // If button exists, ensure it's visible
          ensureButtonVisible();
        }
      }, 300); // Small delay to let page content load
    }
  }, 500); // Check every 500ms for navigation

  // Handle popstate events for browser back/forward buttons
  window.addEventListener('popstate', function(event) {
    console.log('[TranslateButton] Popstate event detected, updating content tracking');
    setTimeout(() => {
      updateContentTracking();
      // Ensure the button is mounted on the new page after back/forward navigation
      if (!document.getElementById('translate-btn')) {
        mountTranslateButton();
      } else {
        // If button exists, ensure it's visible
        ensureButtonVisible();
      }
    }, 100); // Small delay to let page content load
  });

  // Listen for Docusaurus-specific events if available
  if (window.addEventListener) {
    window.addEventListener('load', function() {
      setTimeout(ensureButtonVisible, 500);
    });

    // Listen for custom events that might indicate page changes
    window.addEventListener('DOMContentLoaded', function() {
      setTimeout(ensureButtonVisible, 500);
    });
  }
})();
