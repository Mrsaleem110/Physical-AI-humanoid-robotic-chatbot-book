const React = require('react');
const { render, screen, waitFor } = require('@testing-library/react');
require('@testing-library/jest-dom');

// Mock the useTranslation hook
jest.mock('../../../hooks/useTranslation', () => ({
  useTranslation: jest.fn(() => ({
    currentLanguage: 'en',
    translate: jest.fn().mockImplementation((text, targetLang) => {
      if (targetLang === 'fr') {
        return Promise.resolve(`Translated: ${text}`);
      }
      return Promise.resolve(text);
    }),
    isLoading: false
  })),
}));

// Import the component after mocking dependencies
const TranslateText = require('../TranslateText').default;

describe('TranslateText Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders children as-is when language is English', () => {
    const { getByText } = render(React.createElement(TranslateText, null, 'Hello World'));

    expect(getByText('Hello World')).toBeInTheDocument();
  });

  test('handles translation when language is not English', async () => {
    // Mock useTranslation to return French as current language
    require('../../../hooks/useTranslation').useTranslation.mockReturnValue({
      currentLanguage: 'fr',
      translate: jest.fn().mockResolvedValue('Translated: Hello World'),
      isLoading: false
    });

    render(React.createElement(TranslateText, null, 'Hello World'));

    // Initially shows original text
    expect(screen.getByText('Hello World')).toBeInTheDocument();
  });

  test('renders empty string for empty input', () => {
    const { getByText } = render(React.createElement(TranslateText, null, ''));

    expect(getByText('')).toBeInTheDocument();
  });
});