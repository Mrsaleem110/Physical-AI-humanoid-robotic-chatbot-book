import React from 'react';

// Absolutely minimal translation navbar button that cannot fail
const TranslationNavbarButton = () => {
  return (
    <div className="navbar__item">
      <div
        className="navbar__link"
        style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
      >
        <span>🌐</span>
        <span>Translate</span>
      </div>
    </div>
  );
};

export default TranslationNavbarButton;