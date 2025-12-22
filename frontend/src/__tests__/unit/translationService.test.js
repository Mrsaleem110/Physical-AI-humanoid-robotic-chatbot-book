const { default: TranslationService } = require('../../services/translationService');

// Mock the fetch API
global.fetch = jest.fn();

describe('TranslationService', () => {
  let translationService;

  beforeEach(() => {
    jest.clearAllMocks();
    translationService = new TranslationService();
  });

  test('constructor sets default LibreTranslate URL', () => {
    expect(translationService.libreTranslateUrl).toBe('https://libretranslate.de');
  });

  test('translateText successfully translates text', async () => {
    const mockResponse = { translatedText: 'Bonjour le monde' };
    fetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const result = await translationService.translateText('Hello world', 'fr');

    expect(result).toBe('Bonjour le monde');
    expect(fetch).toHaveBeenCalledWith('https://libretranslate.de/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        q: 'Hello world',
        source: 'auto',
        target: 'fr',
        format: 'text'
      })
    });
  });

  test('translateText handles alternative response format', async () => {
    const mockResponse = [{ translatedText: 'Hola mundo' }];
    fetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const result = await translationService.translateText('Hello world', 'es');

    expect(result).toBe('Hola mundo');
  });

  test('translateText returns original text when API returns error', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });

    const result = await translationService.translateText('Hello world', 'fr');

    expect(result).toBe('Hello world');
  });

  test('translateText returns original text for empty input', async () => {
    const result = await translationService.translateText('', 'fr');

    expect(result).toBe('');
  });

  test('translateText handles API errors gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('Network error'));

    const result = await translationService.translateText('Hello world', 'fr');

    expect(result).toBe('Hello world');
  });

  test('getSupportedLanguages returns language list', async () => {
    const mockLanguages = [
      { code: 'en', name: 'English' },
      { code: 'fr', name: 'French' },
    ];
    fetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockLanguages),
    });

    const result = await translationService.getSupportedLanguages();

    expect(result).toEqual([
      { code: 'en', name: 'English' },
      { code: 'fr', name: 'French' },
    ]);
  });

  test('getSupportedLanguages returns default languages when API fails', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });

    const result = await translationService.getSupportedLanguages();

    expect(result).toHaveLength(8); // Default languages
    expect(result).toContainEqual({ code: 'en', name: 'English' });
    expect(result).toContainEqual({ code: 'ur', name: 'Urdu' });
  });

  test('detectLanguage successfully detects language', async () => {
    const mockResponse = [{ language: 'fr' }];
    fetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    const result = await translationService.detectLanguage('Bonjour le monde');

    expect(result).toBe('fr');
  });

  test('detectLanguage returns default language when API fails', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });

    const result = await translationService.detectLanguage('Bonjour le monde');

    expect(result).toBe('en');
  });
});

describe('TranslationService with API Key', () => {
  test('constructor uses API key from environment', () => {
    // Save original env
    const originalEnv = process.env.REACT_APP_LIBRETRANSLATE_API_KEY;
    process.env.REACT_APP_LIBRETRANSLATE_API_KEY = 'test-api-key';

    const service = new TranslationService();
    expect(service.apiKey).toBe('test-api-key');

    // Restore original env
    process.env.REACT_APP_LIBRETRANSLATE_API_KEY = originalEnv;
  });

  test('translateText includes API key in request when available', async () => {
    // Save original env
    const originalEnv = process.env.REACT_APP_LIBRETRANSLATE_API_KEY;
    process.env.REACT_APP_LIBRETRANSLATE_API_KEY = 'test-api-key';

    const service = new TranslationService();
    const mockResponse = { translatedText: 'Bonjour le monde' };
    fetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResponse),
    });

    await service.translateText('Hello world', 'fr');

    expect(fetch).toHaveBeenCalledWith('https://libretranslate.de/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        q: 'Hello world',
        source: 'auto',
        target: 'fr',
        format: 'text',
        api_key: 'test-api-key'
      })
    });

    // Restore original env
    process.env.REACT_APP_LIBRETRANSLATE_API_KEY = originalEnv;
  });
});