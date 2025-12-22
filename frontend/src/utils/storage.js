// Storage utility for language preferences and translation caching
const LANGUAGE_PREFERENCE_KEY = 'translation_language_preference';
const TRANSLATION_CACHE_KEY = 'translation_cache';

// Simple in-memory cache with size limit
class TranslationCache {
  constructor(maxSize = 100) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  get(key) {
    return this.cache.get(key);
  }

  set(key, value) {
    if (this.cache.size >= this.maxSize) {
      // Remove oldest item (least recently added)
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  clear() {
    this.cache.clear();
  }
}

// Initialize cache
export const TRANSLATION_CACHE = new TranslationCache();

/**
 * Get user's language preference from localStorage
 * @returns {string} Selected language code or default 'en'
 */
export function getLanguagePreference() {
  try {
    const saved = localStorage.getItem(LANGUAGE_PREFERENCE_KEY);
    return saved || 'en';
  } catch (error) {
    console.error('Error reading language preference:', error);
    return 'en'; // Default to English
  }
}

/**
 * Save user's language preference to localStorage
 * @param {string} languageCode - Language code to save
 */
export function setLanguagePreference(languageCode) {
  try {
    localStorage.setItem(LANGUAGE_PREFERENCE_KEY, languageCode);
  } catch (error) {
    console.error('Error saving language preference:', error);
  }
}

/**
 * Get cached translation
 * @param {string} key - Cache key
 * @returns {string|undefined} Cached translation or undefined
 */
export function getCachedTranslation(key) {
  try {
    const cache = JSON.parse(localStorage.getItem(TRANSLATION_CACHE_KEY) || '{}');
    return cache[key];
  } catch (error) {
    console.error('Error reading translation cache:', error);
    return undefined;
  }
}

/**
 * Save translation to cache
 * @param {string} key - Cache key
 * @param {string} value - Translated text
 */
export function setCachedTranslation(key, value) {
  try {
    const cache = JSON.parse(localStorage.getItem(TRANSLATION_CACHE_KEY) || '{}');
    // Limit cache size to prevent storage issues
    const keys = Object.keys(cache);
    if (keys.length > 100) {
      // Remove oldest entries (first 10)
      for (let i = 0; i < 10; i++) {
        if (keys[i]) delete cache[keys[i]];
      }
    }
    cache[key] = value;
    localStorage.setItem(TRANSLATION_CACHE_KEY, JSON.stringify(cache));
  } catch (error) {
    console.error('Error saving translation cache:', error);
  }
}

/**
 * Clear translation cache
 */
export function clearTranslationCache() {
  try {
    localStorage.removeItem(TRANSLATION_CACHE_KEY);
    TRANSLATION_CACHE.clear();
  } catch (error) {
    console.error('Error clearing translation cache:', error);
  }
}