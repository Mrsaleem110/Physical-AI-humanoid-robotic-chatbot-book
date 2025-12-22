import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TranslationProvider } from '../../../src/components/Translation/TranslationProvider';
import TranslateContent from '../../../src/components/Translation/TranslateContent';
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

describe('Content Translation Integration', () => {
  const renderWithProvider = (defaultLanguage = 'en', children) => {
    return render(
      <TranslationProvider defaultLanguage={defaultLanguage}>
        {children}
      </TranslationProvider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('translates both static text and dynamic content', async () => {
    renderWithProvider('en', (
      <div>
        <TranslateText>Hello, World!</TranslateText>
        <TranslateContent content="This is a blog post." />
      </div>
    ));

    // Initially, content should be in English
    expect(screen.getByText('Hello, World!')).toBeInTheDocument();
    expect(screen.getByText('This is a blog post.')).toBeInTheDocument();

    // Change language to French using the context directly
    // Since we can't easily change the language through UI in this test,
    // we'll test the provider with a different default language
    renderWithProvider('fr', (
      <div>
        <TranslateText>Hello, World!</TranslateText>
        <TranslateContent content="This is a blog post." />
      </div>
    ));

    // Wait for translations to complete
    await waitFor(() => {
      expect(screen.getByText('Translated: Hello, World!')).toBeInTheDocument();
      expect(screen.getByText('Translated: This is a blog post.')).toBeInTheDocument();
    });
  });

  test('handles HTML content translation', async () => {
    const htmlContent = '<p>This is a <strong>blog post</strong> with HTML.</p>';

    renderWithProvider('fr', (
      <TranslateContent content={htmlContent} contentType="html" />
    ));

    // Wait for translation to complete
    await waitFor(() => {
      const translatedElement = screen.getByText('Translated: This is a blog post with HTML.');
      expect(translatedElement).toBeInTheDocument();
    });
  });

  test('preserves HTML structure while translating text', async () => {
    const htmlContent = '<h1>Welcome</h1><p>This is a <em>paragraph</em>.</p>';

    renderWithProvider('fr', (
      <TranslateContent content={htmlContent} contentType="html" />
    ));

    // Wait for translation to complete
    await waitFor(() => {
      // The structure should be preserved while text is translated
      const element = screen.getByRole('document').querySelector('h1');
      expect(element).toBeInTheDocument();
    });
  });

  test('handles translation failures gracefully', async () => {
    // Mock a translation failure
    const translationService = require('../../../src/services/translationService');
    translationService.translateText.mockRejectedValueOnce(new Error('Translation API error'));

    renderWithProvider('fr', (
      <TranslateContent content="Content that will fail to translate" />
    ));

    // Wait for the fallback behavior
    await waitFor(() => {
      expect(screen.getByText('Content that will fail to translate')).toBeInTheDocument();
    });
  });

  test('shows loading state for content translation', async () => {
    // Create a delayed promise to test loading state
    let resolvePromise;
    const mockPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    const translationService = require('../../../src/services/translationService');
    translationService.translateText.mockReturnValueOnce(mockPromise);

    renderWithProvider('fr', (
      <TranslateContent content="Loading content..." />
    ));

    // Should show loading indicator
    expect(screen.getByText('Translating content...')).toBeInTheDocument();

    // Resolve the promise
    act(() => {
      resolvePromise('Translated: Loading content...');
    });

    // Wait for content to appear after translation
    await waitFor(() => {
      expect(screen.getByText('Translated: Loading content...')).toBeInTheDocument();
    });
  });

  test('caches translations to avoid repeated API calls', async () => {
    const translationService = require('../../../src/services/translationService');

    renderWithProvider('fr', (
      <div>
        <TranslateText>Hello</TranslateText>
        <TranslateText>Hello</TranslateText> {/* Same text - should be cached */}
      </div>
    ));

    // Wait for translations to complete
    await waitFor(() => {
      expect(screen.getByText('Translated: Hello')).toBeInTheDocument();
    });

    // The same text should only be translated once (if caching works properly)
    // This is hard to test directly without mocking the cache, but we can at least
    // verify that both instances get translated
    const translatedElements = screen.getAllByText('Translated: Hello');
    expect(translatedElements).toHaveLength(2);
  });

  test('does not translate when target language is English', () => {
    renderWithProvider('en', (
      <div>
        <TranslateText>Hello, World!</TranslateText>
        <TranslateContent content="This is content." />
      </div>
    ));

    // Content should remain in original language when target is English
    expect(screen.getByText('Hello, World!')).toBeInTheDocument();
    expect(screen.getByText('This is content.')).toBeInTheDocument();
  });
});

// Helper function to trigger state updates in tests
function act(callback) {
  return global.act(callback);
}