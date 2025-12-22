import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TranslationProvider } from '../../../src/components/Translation/TranslationProvider';
import LanguageSelector from '../../../src/components/Translation/LanguageSelector';
import TranslateText from '../../../src/components/Translation/TranslateText';

// Mock the translation service
jest.mock('../../../src/services/translationService', () => ({
  translateText: jest.fn().mockImplementation((text, targetLang) => {
    if (targetLang === 'fr') {
      return Promise.resolve(`Translated: ${text}`);
    }
    return Promise.resolve(text);
  }),
}));

describe('Language Selection Integration', () => {
  const renderWithProvider = (defaultLanguage = 'en') => {
    return render(
      <TranslationProvider defaultLanguage={defaultLanguage}>
        <div>
          <LanguageSelector />
          <TranslateText>Hello, World!</TranslateText>
        </div>
      </TranslationProvider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('language change triggers content translation', async () => {
    renderWithProvider();

    // Initially, the text should be in English
    expect(screen.getByText('Hello, World!')).toBeInTheDocument();

    // Change language to French
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'fr' } });

    // Wait for the translation to complete
    await waitFor(() => {
      expect(screen.getByText('Translated: Hello, World!')).toBeInTheDocument();
    });
  });

  test('language selection persists across component updates', async () => {
    const { rerender } = renderWithProvider();

    // Change language to French
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'fr' } });

    await waitFor(() => {
      expect(selectElement.value).toBe('fr');
    });

    // Rerender the component (simulating a page update)
    rerender(
      <TranslationProvider defaultLanguage="en">
        <div>
          <LanguageSelector />
          <TranslateText>New content</TranslateText>
        </div>
      </TranslationProvider>
    );

    // The language should remain French
    expect(screen.getByRole('combobox').value).toBe('fr');
  });

  test('translation fails gracefully with fallback to original text', async () => {
    // Mock a translation failure
    const translationService = require('../../../src/services/translationService');
    translationService.translateText.mockRejectedValueOnce(new Error('Translation API error'));

    renderWithProvider();

    // Change language to French (which will trigger a failed translation)
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'fr' } });

    // Wait for the fallback behavior
    await waitFor(() => {
      expect(screen.getByText('Hello, World!')).toBeInTheDocument();
    });
  });

  test('loading state is properly displayed during translation', async () => {
    // Create a promise that doesn't resolve immediately to simulate loading
    let resolvePromise;
    const mockPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    const translationService = require('../../../src/services/translationService');
    translationService.translateText.mockReturnValue(mockPromise);

    renderWithProvider();

    // Change language to trigger loading state
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'fr' } });

    // Translation component should show loading indicator
    expect(screen.getByText('...')).toBeInTheDocument();

    // Resolve the promise to complete the test
    resolvePromise('Translated: Hello, World!');
  });
});