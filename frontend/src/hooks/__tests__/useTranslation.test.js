import { renderHook, act } from '@testing-library/react';
import { TranslationProvider } from '../../components/Translation/TranslationProvider';
import { useTranslation, useTranslateText } from '../useTranslation';

// Mock the translation service
jest.mock('../../services/translationService', () => ({
  translateText: jest.fn().mockImplementation((text, targetLang) => {
    if (targetLang === 'fr') {
      return Promise.resolve(`Translated: ${text}`);
    }
    return Promise.resolve(text);
  }),
}));

const wrapper = ({ children }) => (
  <TranslationProvider defaultLanguage="en">
    {children}
  </TranslationProvider>
);

describe('useTranslation Hook', () => {
  test('provides translation context values', () => {
    const { result } = renderHook(() => useTranslation(), { wrapper });

    expect(result.current).toHaveProperty('currentLanguage');
    expect(result.current).toHaveProperty('availableLanguages');
    expect(result.current).toHaveProperty('isLoading');
    expect(result.current).toHaveProperty('changeLanguage');
    expect(result.current).toHaveProperty('translate');
    expect(result.current).toHaveProperty('getLanguageName');
  });

  test('initializes with default language', () => {
    const { result } = renderHook(() => useTranslation(), { wrapper });

    expect(result.current.currentLanguage).toBe('en');
  });

  test('changeLanguage function updates the language', async () => {
    const { result } = renderHook(() => useTranslation(), { wrapper });

    act(() => {
      result.current.changeLanguage('fr');
    });

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(result.current.currentLanguage).toBe('fr');
  });

  test('translate function calls translation service', async () => {
    const { result } = renderHook(() => useTranslation(), { wrapper });

    let translatedText;
    await act(async () => {
      translatedText = await result.current.translate('Hello', 'fr');
    });

    expect(translatedText).toBe('Translated: Hello');
  });

  test('getLanguageName returns correct language name', () => {
    const { result } = renderHook(() => useTranslation(), { wrapper });

    const languageName = result.current.getLanguageName('fr');
    expect(languageName).toBe('French');
  });

  test('getLanguageName returns code for unknown language', () => {
    const { result } = renderHook(() => useTranslation(), { wrapper });

    const languageName = result.current.getLanguageName('xx');
    expect(languageName).toBe('xx');
  });
});

describe('useTranslateText Hook', () => {
  test('translates text when language is not English', async () => {
    const { result, rerender } = renderHook(
      ({ text, language }) => {
        // We need to create a new provider for each rerender to change the language
        return useTranslateText(text);
      },
      {
        initialProps: { text: 'Hello', language: 'en' },
        wrapper: ({ children, language = 'en' }) => (
          <TranslationProvider defaultLanguage={language}>
            {children}
          </TranslationProvider>
        ),
      }
    );

    // Initially, no translation needed for English
    expect(result.current.translatedText).toBe('Hello');

    // Change language to French and rerender
    rerender({ text: 'Hello', language: 'fr' });

    // Wait for translation to complete
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 100));
    });

    expect(result.current.translatedText).toBe('Translated: Hello');
  });

  test('returns original text when language is English', () => {
    const { result } = renderHook(() => useTranslateText('Hello'), { wrapper });

    expect(result.current.translatedText).toBe('Hello');
  });

  test('handles empty text gracefully', () => {
    const { result } = renderHook(() => useTranslateText(''), { wrapper });

    expect(result.current.translatedText).toBe('');
  });

  test('shows loading state during translation', async () => {
    // Create a delayed promise to test loading state
    let resolvePromise;
    const mockPromise = new Promise((resolve) => {
      resolvePromise = resolve;
    });

    const translationService = require('../../services/translationService');
    translationService.translateText.mockReturnValueOnce(mockPromise);

    const { result } = renderHook(() => useTranslateText('Hello'), {
      wrapper: ({ children }) => (
        <TranslationProvider defaultLanguage="fr">
          {children}
        </TranslationProvider>
      ),
    });

    // Initially loading should be true
    expect(result.current.isLoading).toBe(true);

    // Resolve the promise
    act(() => {
      resolvePromise('Translated: Hello');
    });

    // Wait for state update
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.translatedText).toBe('Translated: Hello');
  });
});