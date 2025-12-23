// Mount TranslateButton to navbar after page loads
(function() {
  console.log('[TranslateButton] Script loaded');
  
  let attempts = 0;
  const maxAttempts = 20;

  // Global API function
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

      const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=en|ur`;
      const response = await fetch(url);
      const data = await response.json();
      
      if (data && data.responseData && data.responseData.translatedText) {
        return data.responseData.translatedText;
      }
      
      return text;
    } catch (e) {
      console.error('[TranslateButton] API error:', e);
      return text;
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
          style="display: flex; align-items: center; justify-content: center; gap: 0.75rem; padding: 0.875rem 1.5rem; font-size: 0.95rem; font-weight: 600; color: white; background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); border: none; border-radius: 8px; cursor: pointer; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 1px 3px rgba(37, 99, 235, 0.2), 0 1px 2px rgba(37, 99, 235, 0.12);"
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

    const button = document.getElementById('translate-btn');
    if (button) {
      setupButton(button);
      console.log('[TranslateButton] Mount complete');
    }
  }

  function setupButton(button) {
    let isTranslated = false;
    let originalContent = null;
    let translatedContent = null;
    let isTranslating = false;

    console.log('[TranslateButton] Setting up button functionality');

    // Check for saved preference
    const preferUrdu = localStorage.getItem('preferUrdu') === 'true';
    if (preferUrdu) {
      const cached = sessionStorage.getItem('translation_' + window.location.pathname);
      if (cached) {
        console.log('[TranslateButton] Restoring cached translation');
        isTranslated = true;
        translatedContent = cached;
        applyTranslation(cached);
      }
    }

    button.addEventListener('click', async (e) => {
      e.preventDefault();
      console.log('[TranslateButton] Button clicked');
      
      if (isTranslating) {
        console.log('[TranslateButton] Already translating, ignoring click');
        return;
      }

      if (isTranslated) {
        restoreOriginal();
      } else {
        await handleTranslate();
      }
    });

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
        if (el) {
          console.log('[TranslateButton] Found content with selector: ' + selector);
          return el;
        }
      }

      console.warn('[TranslateButton] Could not find content element');
      return null;
    }

    async function handleTranslate() {
      isTranslating = true;
      button.disabled = true;
      button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">⏳</span><span class="translate-text">Translating...</span>';

      try {
        const contentElement = getMainContent();
        if (!contentElement) {
          throw new Error('Content element not found on page');
        }

        console.log('[TranslateButton] Found content element, starting translation');
        originalContent = contentElement.innerHTML;

        if (!translatedContent) {
          console.log('[TranslateButton] Calling translateContent');
          translatedContent = await translateContent(originalContent);
          sessionStorage.setItem('translation_' + window.location.pathname, translatedContent);
        }

        applyTranslation(translatedContent);
        isTranslated = true;
        localStorage.setItem('preferUrdu', 'true');
        button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇬🇧</span><span class="translate-text">Show Original</span>';
        button.classList.add('active');
        console.log('[TranslateButton] Translation applied successfully');

      } catch (error) {
        console.error('[TranslateButton] Error:', error);
        alert('Translation Error: ' + error.message);
        button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Translate to Urdu</span>';
      } finally {
        isTranslating = false;
        button.disabled = false;
      }
    }

    function applyTranslation(html) {
      const contentElement = getMainContent();
      if (contentElement) {
        console.log('[TranslateButton] Applying translated HTML');
        contentElement.innerHTML = html;
        contentElement.classList.add('urdu-mode');
        addUrduStyles();
      }
    }

    function restoreOriginal() {
      const contentElement = getMainContent();
      if (contentElement && originalContent) {
        console.log('[TranslateButton] Restoring original content');
        contentElement.innerHTML = originalContent;
        contentElement.classList.remove('urdu-mode');
        isTranslated = false;
        localStorage.setItem('preferUrdu', 'false');
        button.innerHTML = '<span class="translate-icon" style="font-size: 1.25rem;">🇵🇰</span><span class="translate-text">Translate to Urdu</span>';
        button.classList.remove('active');
        removeUrduStyles();
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
        .translate-button.active {
          background: linear-gradient(135deg, #059669 0%, #047857 100%);
        }
      `;
      document.head.appendChild(style);
    }

    function removeUrduStyles() {
      const style = document.getElementById('urdu-styles');
      if (style) style.remove();
    }
  }

  // Start mounting
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountTranslateButton);
  } else {
    mountTranslateButton();
  }
  
  // Also try mounting immediately in case DOM is ready
  setTimeout(mountTranslateButton, 100);
})();
