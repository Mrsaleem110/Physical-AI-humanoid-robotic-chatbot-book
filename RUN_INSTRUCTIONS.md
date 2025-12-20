# Running Your Physical AI & Humanoid Robotics Book with Signup and Urdu Translation

## System Overview
Your website already has complete implementation of both requested features:
1. **User signup system** - Complete with authentication and user profile management
2. **Urdu translation functionality** - Using the free deep-translator service

## How to Run the System

### Step 1: Start the Backend Server
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
The backend will start on `http://localhost:8000`

### Step 2: Start the Frontend Server
In a new terminal window:
```bash
cd frontend
npm install
npm start
```
The frontend will start on `http://localhost:3000`

### Step 3: Access Your Website
1. Open your browser to `http://localhost:3000`
2. If you're not logged in, you'll be redirected to the signup page (`/auth`)
3. If you're logged in, you'll see the homepage with all features
4. Create an account or login with existing credentials

## Translation Features (Completely Free!)

Your system uses the **deep-translator library** which provides free translation without API keys:
- No monthly charges or usage limits
- Supports Urdu and many other languages
- Already configured in your backend

## How to Use Translation

1. **Language Switcher**: Click the globe icon (🌐) in the top-right navbar to switch between English and Urdu
2. **Page Translation**: Use the translation button on individual pages to translate content in real-time
3. **Book Content**: Chapters and book content will be translated when you switch languages

## Key Features Already Implemented

### Authentication System:
- ✅ Complete signup/login flow
- ✅ User profile collection with background questionnaire
- ✅ Automatic redirect to signup for non-authenticated users
- ✅ Secure session management

### Translation System:
- ✅ Free Urdu translation using deep-translator
- ✅ Language switcher with RTL support for Urdu
- ✅ Real-time page translation
- ✅ Book content translation
- ✅ Proper handling of Urdu right-to-left text direction

## No API Keys Required!
Your system is already configured to use the free deep-translator service, so you don't need to get any API keys.

## Troubleshooting

1. **If translation isn't working**:
   - Make sure both backend (port 8000) and frontend (port 3000) are running
   - Check that the backend translation service is initialized (check console logs)

2. **If signup isn't working**:
   - Verify the backend server is running
   - Check that the authentication endpoints are accessible

3. **If language switching isn't working**:
   - Refresh the page after changing languages
   - Check browser console for any JavaScript errors

## Security Note
All sensitive information like API keys, database URLs, etc. are properly stored in environment files and not exposed in the frontend code.

Your website is ready to use with both signup and Urdu translation features completely free of charge!