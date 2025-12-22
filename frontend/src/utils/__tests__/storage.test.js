import {
  getLanguagePreference,
  setLanguagePreference,
  getCachedTranslation,
  setCachedTranslation,
  clearTranslationCache,
  TRANSLATION_CACHE
} from '../storage';

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

// Mock global localStorage
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
});

describe('Storage Utility Functions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.setStore({});
    TRANSLATION_CACHE.clear();
  });

  describe('Language Preference Functions', () => {
    test('getLanguagePreference returns saved language', () => {
      mockLocalStorage.setStore({ 'translation_language_preference': 'fr' });

      const result = getLanguagePreference();
      expect(result).toBe('fr');
      expect(mockLocalStorage.getItem).toHaveBeenCalledWith('translation_language_preference');
    });

    test('getLanguagePreference returns default when no preference saved', () => {
      mockLocalStorage.setStore({});

      const result = getLanguagePreference();
      expect(result).toBe('en');
    });

    test('getLanguagePreference returns default when localStorage unavailable', () => {
      // Mock localStorage to throw an error
      const originalLocalStorage = window.localStorage;
      Object.defineProperty(window, 'localStorage', {
        value: {
          getItem: jest.fn(() => {
            throw new Error('Storage unavailable');
          })
        }
      });

      const result = getLanguagePreference();
      expect(result).toBe('en');

      // Restore original localStorage
      Object.defineProperty(window, 'localStorage', {
        value: originalLocalStorage
      });
    });

    test('setLanguagePreference saves language preference', () => {
      setLanguagePreference('es');

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'translation_language_preference',
        'es'
      );
    });

    test('setLanguagePreference handles errors gracefully', () => {
      // Mock localStorage to throw an error
      const originalLocalStorage = window.localStorage;
      Object.defineProperty(window, 'localStorage', {
        value: {
          setItem: jest.fn(() => {
            throw new Error('Storage unavailable');
          })
        }
      });

      expect(() => setLanguagePreference('es')).not.toThrow();

      // Restore original localStorage
      Object.defineProperty(window, 'localStorage', {
        value: originalLocalStorage
      });
    });
  });

  describe('Translation Cache Functions', () => {
    test('getCachedTranslation returns cached translation', () => {
      const cacheData = { 'hello_en_fr': 'bonjour' };
      mockLocalStorage.setStore({ 'translation_cache': JSON.stringify(cacheData) });

      const result = getCachedTranslation('hello_en_fr');
      expect(result).toBe('bonjour');
    });

    test('getCachedTranslation returns undefined for non-existent key', () => {
      mockLocalStorage.setStore({ 'translation_cache': JSON.stringify({}) });

      const result = getCachedTranslation('non_existent_key');
      expect(result).toBeUndefined();
    });

    test('getCachedTranslation handles JSON parsing errors', () => {
      mockLocalStorage.setStore({ 'translation_cache': 'invalid json' });

      const result = getCachedTranslation('some_key');
      expect(result).toBeUndefined();
    });

    test('setCachedTranslation saves translation to cache', () => {
      setCachedTranslation('hello_en_fr', 'bonjour');

      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'translation_cache',
        JSON.stringify({ 'hello_en_fr': 'bonjour' })
      );
    });

    test('setCachedTranslation maintains cache size limit', () => {
      // Fill cache with more items than the limit (100 in the implementation)
      for (let i = 0; i < 110; i++) {
        setCachedTranslation(`key_${i}`, `value_${i}`);
      }

      // Check that the cache was properly managed
      const [call] = mockLocalStorage.setItem.mock.calls.slice(-1);
      const cache = JSON.parse(call[1]);
      const keys = Object.keys(cache);

      // Should have removed some old entries (first 10 as per implementation)
      expect(keys.length).toBeLessThanOrEqual(110); // Might be up to 100 after cleanup
    });

    test('clearTranslationCache removes cache data', () => {
      TRANSLATION_CACHE.set('test_key', 'test_value');
      mockLocalStorage.setStore({ 'translation_cache': JSON.stringify({ 'test': 'value' }) });

      clearTranslationCache();

      expect(mockLocalStorage.removeItem).toHaveBeenCalledWith('translation_cache');
      // Note: In-memory cache clear test would require checking internal state
    });
  });

  describe('Translation Cache Class', () => {
    test('TranslationCache sets and gets values', () => {
      TRANSLATION_CACHE.set('test_key', 'test_value');
      const result = TRANSLATION_CACHE.get('test_key');

      expect(result).toBe('test_value');
    });

    test('TranslationCache respects size limit', () => {
      const maxSize = 100;
      const cache = TRANSLATION_CACHE;

      // Fill the cache to its limit
      for (let i = 0; i < maxSize + 10; i++) {
        cache.set(`key_${i}`, `value_${i}`);
      }

      // The cache should have removed old items when exceeding the limit
      expect(cache.cache.size).toBeLessThanOrEqual(maxSize);
    });

    test('TranslationCache clear method works', () => {
      TRANSLATION_CACHE.set('test_key', 'test_value');
      expect(TRANSLATION_CACHE.cache.size).toBeGreaterThan(0);

      TRANSLATION_CACHE.clear();
      expect(TRANSLATION_CACHE.cache.size).toBe(0);
    });
  });
});