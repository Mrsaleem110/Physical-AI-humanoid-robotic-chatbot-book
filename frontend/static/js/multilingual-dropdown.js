// Multilingual dropdown functionality

// Ensure translation configuration is available
if (typeof window.translationConfig === 'undefined') {
  window.translationConfig = {
    apiUrl: 'http://localhost:5000/translate'
  };
}

// Wait for the DOM to be loaded
document.addEventListener('DOMContentLoaded', function() {
  // Create the dropdown element
  const dropdownContainer = document.getElementById('multilingual-dropdown-container');
  if (!dropdownContainer) return;

  // Supported languages with their native names and codes
  const languages = [
    { code: 'en', name: 'English', native: 'English' },
    { code: 'ur', name: 'Urdu', native: 'اردو' }
  ];

  // Get current language from URL
  const getCurrentLanguage = () => {
    const path = window.location.pathname;
    const pathParts = path.split('/');
    if (pathParts[1] && languages.some(lang => lang.code === pathParts[1])) {
      return pathParts[1];
    }
    return 'en'; // default to English
  };

  // Helper function to get language info
  const getLanguageInfo = (code) => {
    return languages.find(lang => lang.code === code) || { code: code, name: code, native: code };
  };

  const currentLang = getCurrentLanguage();
  const currentLanguageInfo = languages.find(lang => lang.code === currentLang);

  // Create dropdown HTML
  const dropdownHTML = `
    <div class="multilingual-dropdown" style="position: relative; display: inline-block;">
      <button id="multilingual-toggle" style="
        background-color: #007cba;
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: background-color 0.3s ease;
      " onmouseover="this.style.backgroundColor='#005a87'" onmouseout="this.style.backgroundColor='#007cba'">
        <span>🌐</span>
        <span>${currentLanguageInfo?.native || 'English'}</span>
        <span style="margin-left: 4px;">▼</span>
      </button>

      <div id="multilingual-menu" style="
        position: absolute;
        top: 100%;
        right: 0;
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        z-index: 1000;
        min-width: 150px;
        margin-top: 4px;
        display: none;
      ">
        ${languages.map(lang => `
          <div
            class="lang-option"
            data-lang="${lang.code}"
            style="
              padding: 8px 12px;
              cursor: pointer;
              border-bottom: ${lang.code !== languages[languages.length - 1].code ? '1px solid #eee' : 'none'};
              background-color: ${currentLang === lang.code ? '#f5f5f5' : 'white'};
              font-weight: ${currentLang === lang.code ? 'bold' : 'normal'};
            "
            onmouseover="this.style.backgroundColor='${currentLang === lang.code ? '#f5f5f5' : '#f0f0f0'}'"
            onmouseout="this.style.backgroundColor='${currentLang === lang.code ? '#f5f5f5' : 'white'}'"
          >
            ${lang.native} (${lang.name})
          </div>
        `).join('')}
      </div>
    </div>
  `;

  dropdownContainer.innerHTML = dropdownHTML;

  // Add event listeners
  const toggleButton = document.getElementById('multilingual-toggle');
  const menu = document.getElementById('multilingual-menu');
  const langOptions = document.querySelectorAll('.lang-option');

  let isOpen = false;

  // Toggle menu
  toggleButton.addEventListener('click', function(e) {
    e.stopPropagation();
    isOpen = !isOpen;
    menu.style.display = isOpen ? 'block' : 'none';

    // Update arrow
    const arrow = this.querySelector('span:last-child');
    arrow.textContent = isOpen ? '▲' : '▼';
  });

  // Handle language selection with user control
  langOptions.forEach(option => {
    option.addEventListener('click', function() {
      const langCode = this.getAttribute('data-lang');
      const currentHost = window.location.origin;
      const currentLang = getCurrentLanguage();


      // Navigate to the selected language
      // Try to maintain the current page if it exists in the target language
      const currentPath = window.location.pathname;
      let targetPath;

      // Remove current language prefix if present
      const pathParts = currentPath.split('/');
      if (pathParts[1] && languages.some(lang => lang.code === pathParts[1])) {
        targetPath = '/' + pathParts.slice(2).join('/');
      } else {
        targetPath = currentPath;
      }

      // If we're at root or the target path is empty, go to the language root
      if (targetPath === '/' || targetPath === '') {
        window.location.href = currentHost + '/' + langCode + '/';
      } else {
        window.location.href = currentHost + '/' + langCode + targetPath;
      }
    });
  });

  // Close menu when clicking outside
  document.addEventListener('click', function(e) {
    if (!dropdownContainer.contains(e.target)) {
      menu.style.display = 'none';
      isOpen = false;
      const arrow = toggleButton.querySelector('span:last-child');
      arrow.textContent = '▼';
    }
  });

  // Update dropdown when language changes (for back/forward buttons)
  window.addEventListener('popstate', function() {
    const newCurrentLang = getCurrentLanguage();
    const newCurrentLanguageInfo = languages.find(lang => lang.code === newCurrentLang);
    const langButton = toggleButton.querySelector('span:nth-child(2)');
    if (langButton && newCurrentLanguageInfo) {
      langButton.textContent = newCurrentLanguageInfo.native;
    }
  });
});