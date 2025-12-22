Translation feature
===================

This project uses Docusaurus i18n for static translations and a small client-side translation system for dynamic translations.

What I added
- A React `LanguageToggle` component that mounts into the navbar container and toggles between English and Urdu.
- Enhancements to `frontend/static/js/multilingual-dropdown.js` and `frontend/static/js/controlled-translation.js` to persist the user's language choice in `localStorage` (key `preferred_language`) and to dispatch/handle `translate` events.
- A `TranslationContext` that uses LibreTranslate (public instance) with MyMemory fallback and caches translations.

How to run locally
1. From the repository root open a terminal and run:

```bash
cd frontend
npm install
npm run start
```

2. Open the site in your browser. Use the language control in the navbar (left of Sign In) to toggle between English and Urdu. The selection is stored in the browser's `localStorage` and persists across reloads.

Notes on free translation
- The UI persistence uses `localStorage` (free client-side storage). No paid API keys are required.
- The dynamic translation service uses the public LibreTranslate endpoint (https://libretranslate.com) with MyMemory fallback; both offer limited free usage. For reliable self-hosted translation, run a LibreTranslate container and point `window.translationConfig.apiUrl` to it.

Changing default behavior
- To force Docusaurus to serve the localized pages (recommended for Urdu), ensure the `i18n` folder contains translations (already present in `frontend/i18n/ur`). The toggle will attempt to redirect to the language-prefixed URL.

If you'd like, I can:
- Wire an option to use a self-hosted LibreTranslate instance (help with docker-compose). 
- Add per-page translated JSON resources or expand Urdu translations.

Self-hosted LibreTranslate (recommended)
-------------------------------------
You already have a `docker-compose.yml` at the project root that includes a `libretranslate` service. To run it:

```bash
# from repository root
docker-compose up -d libretranslate
```

This will start LibreTranslate on port `5000` by default. The app is already configured to prefer a local API URL: `window.translationConfig.apiUrl` or the Docusaurus `customFields.libreTranslateUrl` (set in `docusaurus.config.js`). The default points to `http://localhost:5000/translate`.

After starting the service, restart the frontend dev server (`npm run start`) and the dynamic translation requests will go to your local LibreTranslate instance (free to run).
