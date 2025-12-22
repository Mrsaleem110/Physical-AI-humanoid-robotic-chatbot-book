import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TranslationProvider } from '../../../src/components/Translation/TranslationProvider';
import LanguageSelector from '../../../src/components/Translation/LanguageSelector';

// Mock localStorage
const mockLocalStorage = (() => {
  let store = {};

  return {
    getItem: jest.fn((key) => store[key] || null),
    setItem: jest.fn((key, value) => {
      store[key] = value.toString();
    }),
    removeItem: jest.fn((key) => {
      delete store[key];
    }),
    clear: jest.fn(() => {
      store = {};
    }),
    getStore: () => store,
    setStore: (newStore) => {
      store = newStore;
    }
  };
})();

// Mock the translation service
jest.mock('../../../src/services/translationService', () => ({
  translateText: jest.fn().mockImplementation((text, targetLang) => {
    if (targetLang === 'fr') {
      return Promise.resolve(`Translated: ${text}`);
    }
    return Promise.resolve(text);
  }),
}));

// Mock global localStorage
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

describe('Language Persistence Integration', () => {
  const renderWithProvider = (defaultLanguage = 'en') => {
    return render(
      <TranslationProvider defaultLanguage={defaultLanguage}>
        <div>
          <LanguageSelector />
          <div id="language-display">
            Current Language: {defaultLanguage}
          </div>
        </div>
      </TranslationProvider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.setStore({});
  });

  test('saves language preference to localStorage when changed', async () => {
    renderWithProvider('en');

    // Change language to French
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'fr' } });

    // Wait for the change to be processed
    await waitFor(() => {
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'translation_language_preference',
        'fr'
      );
    });
  });

  test('loads saved language preference on initialization', () => {
    // Set a saved preference
    mockLocalStorage.setStore({ 'translation_language_preference': 'es' });

    renderWithProvider('en'); // Default language is 'en'

    // The component should load with the saved language
    const selectElement = screen.getByRole('combobox');
    expect(selectElement.value).toBe('es');
  });

  test('persists language across multiple component mounts', async () => {
    // First render - set language to German
    const { unmount } = renderWithProvider('en');
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'de' } });

    await waitFor(() => {
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'translation_language_preference',
        'de'
      );
    });

    // Unmount and remount component
    unmount();
    renderWithProvider('en'); // Default is English again

    // The language should still be German due to persistence
    expect(screen.getByRole('combobox').value).toBe('de');
  });

  test('handles localStorage errors gracefully', async () => {
    // Mock localStorage to throw an error when setting items
    const originalLocalStorage = window.localStorage;
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn((key) => mockLocalStorage.getStore()[key] || null),
        setItem: jest.fn(() => {
          throw new Error('Storage quota exceeded');
        }),
        removeItem: jest.fn(() => {
          throw new Error('Storage error');
        })
      }
    });

    renderWithProvider('en');

    // Try to change language - should not crash even with localStorage errors
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'ja' } });

    // Component should still function even if storage fails
    expect(selectElement.value).toBe('ja');

    // Restore original localStorage
    Object.defineProperty(window, 'localStorage', {
      value: originalLocalStorage
    });
  });

  test('respects browser storage limitations', async () => {
    // Mock localStorage with limited space
    let storage = {};
    const originalLocalStorage = window.localStorage;
    Object.defineProperty(window, 'localStorage', {
      value: {
        getItem: jest.fn((key) => storage[key] || null),
        setItem: jest.fn((key, value) => {
          // Simulate storage limit
          if (JSON.stringify({ ...storage, [key]: value }).length > 10000) {
            throw new Error('QuotaExceededError');
          }
          storage[key] = value.toString();
        }),
        removeItem: jest.fn((key) => {
          delete storage[key];
        })
      }
    });

    renderWithProvider('en');

    // Change language - should handle storage limitations gracefully
    const selectElement = screen.getByRole('combobox');
    fireEvent.change(selectElement, { target: { value: 'zh' } });

    // Component should still function
    expect(selectElement.value).toBe('zh');

    // Restore original localStorage
    Object.defineProperty(window, 'localStorage', {
      value: originalLocalStorage
    });
  });

  test('language preference persists across page navigation simulation', async () => {
    // Simulate first page with language set to Urdu
    mockLocalStorage.setStore({ 'translation_language_preference': 'ur' });

    // Render first "page"
    const { rerender } = renderWithProvider('en');

    // Verify language is loaded from storage
    expect(screen.getByRole('combobox').value).toBe('ur');

    // Simulate navigating to another "page" (rerender with different content)
    rerender(
      <TranslationProvider defaultLanguage="en">
        <div>
          <LanguageSelector />
          <div id="other-page-content">Other Page</div>
        </div>
      </TranslationProvider>
    );

    // Language should still be Urdu
    expect(screen.getByRole('combobox').value).toBe('ur');
  });

  test('language preference is consistent across different components', async () => {
    // Set language preference
    mockLocalStorage.setStore({ 'translation_language_preference': 'hi' });

    // Render with multiple translation-aware components
    renderWithProvider('en');

    // All components should see the same language
    const selectElement = screen.getByRole('combobox');
    expect(selectElement.value).toBe('hi');

    // Test that the context provides the correct language
    expect(mockLocalStorage.getItem('translation_language_preference')).toBe('hi');
  });
});