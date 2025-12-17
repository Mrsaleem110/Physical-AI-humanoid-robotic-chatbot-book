import React from 'react';
import TranslateButton from '../components/TranslateButton';

// This component wraps the translate button for use in MDX files
const TranslateButtonWrapper = () => {
  return (
    <div style={{
      position: 'sticky',
      top: '10px',
      zIndex: 1000,
      marginBottom: '20px',
      textAlign: 'right'
    }}>
      <TranslateButton />
    </div>
  );
};

export default TranslateButtonWrapper;