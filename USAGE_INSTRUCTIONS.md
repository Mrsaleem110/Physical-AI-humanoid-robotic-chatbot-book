# Usage Instructions: Signup and Urdu Translation Features

## Overview

Your Physical AI & Humanoid Robotics Book website is already fully equipped with:
1. A comprehensive user signup and authentication system
2. Urdu translation functionality with support for multiple translation services

## Signup System

### How it works:
- When unauthenticated users visit your website homepage, they are redirected to the signup/login page (`/auth`)
- Authenticated users see the homepage with all features
- Users can create an account using the signup form with their name, email, and password
- After signup, users go through a profile questionnaire to personalize their learning experience
- Once authenticated, users can access the full book content

### Signup Flow:
1. User visits the homepage
2. Automatically redirected to `/auth` if not logged in
3. User can choose between "Login" or "Sign Up"
4. After signup, users complete a background questionnaire
5. Users are redirected to the homepage with personalized content

## Urdu Translation Features

### Available Translation Options:

#### 1. Language Switcher Dropdown
- Located in the top-right corner of the navbar
- Shows a globe icon 🌐 with the current language
- Allows switching between English and Urdu (with native names: English/اردو)
- Changes the URL structure (e.g., `/ur/` for Urdu content)

#### 2. Page Translation Component
- Located on individual pages/chapters
- Provides a translate button that can translate the entire page content
- Supports Urdu as the target language
- Can toggle between translated and original content

### Translation Services Configuration:

The system supports multiple translation services with fallback options:

1. **Google Translate API** (Most accurate)
   - Add your API key to `.env`: `REACT_APP_GOOGLE_TRANSLATE_API_KEY=your_key_here`

2. **LibreTranslate** (Free, self-hosted)
   - Default configuration: `REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate`
   - Can use public instances or self-hosted service

3. **Fallback System**
   - If all services fail, the system uses mock translations for demonstration

## Setup Instructions

### 1. For Google Translate API (Recommended for Production):
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Translate API
4. Create an API key
5. Update your `.env` file:
   ```
   REACT_APP_GOOGLE_TRANSLATE_API_KEY=your_actual_api_key_here
   ```

### 2. For LibreTranslate (Free Option):
1. Install LibreTranslate:
   ```bash
   pip install libretranslate
   ```
2. Start the service:
   ```bash
   libretranslate --host 0.0.0.0 --port 5000
   ```
   Or use Docker:
   ```bash
   docker run -d -p 5000:5000 --name libretranslate libretranslate/libretranslate
   ```

### 3. Start the Application:
```bash
# Start backend (if needed)
cd backend
python -m uvicorn main:app --reload

# In a new terminal, start frontend
cd frontend
npm install
npm start
```

## Testing the Features

### Test Signup Flow:
1. Navigate to your website homepage
2. You should be redirected to `/auth`
3. Complete the signup form
4. Fill in the background questionnaire
5. Verify you're redirected to the homepage with personalized experience

### Test Urdu Translation:
1. Use the language switcher dropdown to select Urdu
2. Verify the content changes to Urdu (if static translations exist)
3. Use the page translation component to translate dynamic content
4. Test switching back to English

## Environment Variables

Your `.env` file should contain:

```
REACT_APP_BACKEND_API_URL=http://localhost:8000
REACT_APP_GOOGLE_TRANSLATE_API_KEY=your_google_translate_api_key_here  # Optional
REACT_APP_LIBRETRANSLATE_URL=http://localhost:5000/translate          # For free translation
```

## Troubleshooting

### Common Issues:

1. **Translation not working**:
   - Check that your API keys are correctly set in `.env`
   - Verify that translation services are running
   - Restart the development server after changing environment variables

2. **Signup redirect not working**:
   - Ensure the UserContext is properly initialized
   - Check that the authentication API endpoints are accessible

3. **Language switching issues**:
   - Verify that the i18n configuration in `docusaurus.config.js` is correct
   - Check that the MultilingualDropdown component is properly rendered

## Security Notes

- Never commit your `.env` file with actual API keys
- Use environment variables for sensitive information
- Consider using backend proxy for production to hide API keys from client-side code