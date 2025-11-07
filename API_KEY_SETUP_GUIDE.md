# API Key Setup Guide

## Overview
The AI Fitness Hub now includes a user-friendly API key management system that allows users to configure their own Gemini API key for AI-powered features.

## Features Implemented

### 1. **API Key Settings Page** (`pages/ApiKeySettings.tsx`)

#### ✅ **User Interface**
- Clean, intuitive settings page matching the app's blue/sky gradient theme
- API key input field with show/hide toggle (Eye/EyeOff icons)
- Real-time validation feedback with error/success states
- Clear instructions with 5 numbered steps
- Privacy notice assuring local storage only

#### ✅ **API Key Validation**
```typescript
const validateApiKey = async (key: string): Promise<boolean>
```
- **Basic Validation**: Checks length (≥30 characters) and alphanumeric format
- **API Validation**: Makes actual test request to Gemini API to verify key works
- **Endpoint**: `POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${apiKey}`
- **Test Prompt**: "Hello" (minimal request to verify authentication)

#### ✅ **Secure Storage**
- Uses browser's `localStorage` with key: `'gemini_api_key'`
- No server-side storage - completely client-side
- API key stays on user's device
- Export function `getApiKey()` for service access

#### ✅ **User Actions**
- **Save**: Validates and stores API key
- **Clear**: Removes stored key from localStorage
- **Show/Hide**: Toggle visibility for security

### 2. **Header Integration** (`components/Header.tsx`)

- Added Settings button to main navigation
- Icon: Settings (from lucide-react)
- Accessible from all pages
- Consistent styling with other nav buttons

### 3. **Routing Setup** (`App.tsx`)

- Added `'api-settings'` to Page type union
- Created route case for ApiKeySettings component
- Page title: "API Key Settings"
- Page description: "Configure your Gemini API key to enable AI-powered features."

### 4. **Service Integration** (`services/geminiService.ts`)

#### ✅ **All AI Functions Updated**

**getFitnessTips()**
```typescript
const apiKey = getApiKey();
if (!apiKey) {
  return "⚠️ **API Key Required**\n\nTo get AI-powered insights...";
}
const ai = new GoogleGenAI({ apiKey });
```

**getWorkoutAnalysis()**
```typescript
const apiKey = getApiKey();
if (!apiKey) {
  return "⚠️ **API Key Required**\n\nTo get AI-powered workout analysis...";
}
const ai = new GoogleGenAI({ apiKey });
```

**chatWithAI()**
```typescript
const apiKey = getApiKey();
if (!apiKey) {
  return "⚠️ I'd love to chat, but I need an API key first!...";
}
const ai = new GoogleGenAI({ apiKey });
```

## User Workflow

### First-Time Setup
1. User opens app and navigates to any page
2. Clicks "Settings" button in header
3. Follows 5-step instructions to get Gemini API key
4. Enters key in input field
5. Clicks "Save API Key"
6. System validates key (basic + API test)
7. Success message confirms key is saved
8. User returns to app features

### Using AI Features
- **Without API Key**: Friendly message prompts user to configure key in Settings
- **With API Key**: Full AI features work seamlessly
  - Fitness tips on results page
  - Workout analysis after exercise session
  - AI chat for exercise questions

### Managing API Key
- **View Settings**: Shows "✅ API Key Configured" if key exists
- **Clear Key**: One-click removal with confirmation
- **Update Key**: Save new key anytime (overwrites old one)

## How to Get a Gemini API Key

### Step-by-Step Instructions (shown in UI)

1. **Visit Google AI Studio**
   - Go to: https://aistudio.google.com/app/apikey

2. **Sign in with Google Account**
   - Use your Google account credentials

3. **Create API Key**
   - Click "Create API Key" button
   - Select or create a project
   - Copy the generated key

4. **Enter Key in Settings**
   - Paste key into input field
   - Click "Save API Key"

5. **Start Using AI Features**
   - Return to workout or calculator
   - Enjoy AI-powered insights!

## Technical Details

### Storage Schema
```typescript
localStorage.setItem('gemini_api_key', apiKey);
const key = localStorage.getItem('gemini_api_key');
```

### Validation Logic
```typescript
// 1. Basic format check
if (key.length < 30) return false;
if (!/^[a-zA-Z0-9_-]+$/.test(key)) return false;

// 2. API test request
const response = await fetch(
  `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${key}`,
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: 'Hello' }] }]
    })
  }
);

return response.ok; // 200 = valid, 400/401 = invalid
```

### Error Handling
- **Invalid Format**: "Please enter a valid API key (minimum 30 characters)"
- **API Rejection**: "Invalid API key. Please check and try again."
- **Network Error**: Error message with details
- **Missing Key**: Graceful fallback with helpful instructions

## Security & Privacy

### ✅ Client-Side Only
- API key stored in browser's localStorage
- Never sent to any server except Google AI
- No backend storage or logging

### ✅ User Control
- User provides their own API key
- User can clear key anytime
- No hidden usage or charges

### ✅ Transparency
- Privacy notice on settings page
- Clear explanation of where key is stored
- Instructions link to official Google AI Studio

## Benefits

### For Users
- 🔐 Full control over API key
- 💰 Direct billing to their Google account
- 🚀 No dependency on app owner's quota
- 🔒 Privacy - key stays on their device
- ✨ Immediate access to AI features

### For Developers
- 📦 No API key management required
- 💸 No API costs passed to app owner
- 🎯 Scalable - each user uses own quota
- 🛡️ No liability for API abuse
- 🌍 Works for all users globally

## File Changes Summary

### New Files
- ✅ `pages/ApiKeySettings.tsx` (280 lines)
- ✅ `API_KEY_SETUP_GUIDE.md` (this file)

### Modified Files
- ✅ `types.ts` - Added `'api-settings'` to Page type
- ✅ `components/Header.tsx` - Added Settings button
- ✅ `App.tsx` - Added routing for api-settings page
- ✅ `services/geminiService.ts` - Updated all 3 AI functions to use stored key

## Testing Checklist

- [ ] Settings button appears in header
- [ ] Clicking Settings navigates to API key page
- [ ] Page matches blue/sky gradient theme
- [ ] Can enter API key in input field
- [ ] Show/hide toggle works
- [ ] Invalid key shows error message
- [ ] Valid key shows success message
- [ ] Key persists after page refresh
- [ ] Clear button removes key
- [ ] AI features show prompt when no key configured
- [ ] AI features work when key is configured
- [ ] Validation makes actual API request
- [ ] Error handling works for network issues

## Future Enhancements (Optional)

### 1. **API Key Indicator**
- Show icon in header when key is configured
- Visual feedback without entering settings

### 2. **Inline Prompts**
- "Configure API Key" button in AI tip placeholders
- Direct navigation from feature to settings

### 3. **Key Health Check**
- Periodic validation to detect expired/revoked keys
- Alert user if key becomes invalid

### 4. **Usage Tracking**
- Show user their API call count (client-side)
- Help them monitor quota usage

### 5. **Multiple Providers**
- Support for other AI providers (OpenAI, Anthropic)
- User chooses preferred AI service

## Conclusion

The API key management system is now fully integrated and functional. Users can:
1. ✅ Access Settings from header
2. ✅ Enter and validate their Gemini API key
3. ✅ Store key securely in localStorage
4. ✅ Use all AI features with their own key
5. ✅ Manage (view/clear/update) key anytime

The implementation is secure, user-friendly, and follows best practices for client-side API key management. 🎉
