import { Check, Eye, EyeOff, Key, Save, Settings, X } from 'lucide-react';
import { useEffect, useState } from 'react';

const API_KEY_STORAGE_KEY = 'gemini_api_key';

export default function ApiKeySettings() {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const [isValidating, setIsValidating] = useState(false);

  useEffect(() => {
    // Load saved API key
    const savedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
    if (savedKey) {
      setApiKey(savedKey);
      setSaved(true);
    }
  }, []);

  const validateApiKey = (key: string): boolean => {
    // Basic validation: Should start with specific pattern and have minimum length
    const isValid = key.length >= 30 && /^[A-Za-z0-9_-]+$/.test(key);
    return isValid;
  };

  const handleSave = async () => {
    setError('');
    setSaved(false);

    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    if (!validateApiKey(apiKey)) {
      setError('Invalid API key format. Please check your key and try again.');
      return;
    }

    setIsValidating(true);

    // Test API key with a simple request
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${apiKey}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: 'test' }] }],
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('API validation error:', errorData);
        throw new Error('Invalid API key');
      }

      // Save to localStorage
      localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Validation error:', err);
      setError('Failed to validate API key. Please check that your key is correct and has proper permissions.');
    } finally {
      setIsValidating(false);
    }
  };

  const handleClear = () => {
    setApiKey('');
    localStorage.removeItem(API_KEY_STORAGE_KEY);
    setSaved(false);
    setError('');
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="glass-card rounded-2xl shadow-2xl p-8 mb-6 animate-fade-in-scale">
          <div className="flex items-center gap-4 mb-6">
            <div className="bg-gradient-to-br from-slate-900 to-slate-700 p-3 rounded-xl shadow-lg">
              <Settings className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold font-spartan text-slate-900">
                API Settings
              </h1>
              <p className="text-slate-600">Configure your Gemini API key for AI-powered features</p>
            </div>
          </div>

          {/* Info Box */}
          <div className="bg-blue-50/80 border-l-4 border-blue-500 p-4 rounded-lg mb-6 backdrop-blur-sm">
            <div className="flex items-start gap-3">
              <Key className="w-5 h-5 text-blue-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="font-semibold text-blue-900 mb-1">Why do I need an API key?</p>
                <p className="text-blue-800">
                  The AI coaching features use Google's Gemini API. You need to provide your own API key to access these features.
                  Don't have one?{' '}
                  <a
                    href="https://makersuite.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline font-semibold hover:text-blue-600 transition-colors"
                  >
                    Get your free API key here
                  </a>
                </p>
              </div>
            </div>
          </div>

          {/* API Key Input */}
          <div className="space-y-4">
            <label className="block">
              <span className="text-slate-700 font-semibold mb-2 block">Gemini API Key</span>
              <div className="relative">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter your Gemini API key..."
                  className="w-full px-4 py-3 pr-12 border-2 border-slate-300 rounded-xl focus:border-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all outline-none bg-white"
                />
                <button
                  type="button"
                  onClick={() => setShowKey(!showKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-2 hover:bg-slate-100 rounded-lg transition-colors"
                  title={showKey ? 'Hide key' : 'Show key'}
                >
                  {showKey ? (
                    <EyeOff className="w-5 h-5 text-slate-500" />
                  ) : (
                    <Eye className="w-5 h-5 text-slate-500" />
                  )}
                </button>
              </div>
            </label>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-4 rounded-lg flex items-start gap-3">
                <X className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-red-800">{error}</p>
              </div>
            )}

            {/* Success Message */}
            {saved && (
              <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded-lg flex items-start gap-3">
                <Check className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-green-800 font-semibold">
                  API key saved successfully! AI features are now enabled.
                </p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={handleSave}
                disabled={isValidating || !apiKey.trim()}
                className="flex-1 bg-slate-900 hover:bg-blue-600 text-white px-6 py-3 rounded-xl font-semibold transition-all duration-500 shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                <Save className="w-5 h-5" />
                {isValidating ? 'Validating...' : 'Save & Validate'}
              </button>
              <button
                onClick={handleClear}
                className="px-6 py-3 border-2 border-slate-300 text-slate-700 rounded-xl font-semibold hover:bg-slate-50 transition-all"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="glass-card rounded-2xl shadow-2xl p-8 animate-fade-in-scale" style={{ animationDelay: '0.1s' }}>
          <h2 className="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
            <span>📝</span> How to get your API key
          </h2>
          <ol className="space-y-3 text-slate-700">
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-slate-900 text-white rounded-full flex items-center justify-center text-sm font-bold">
                1
              </span>
              <div>
                <span className="font-semibold">Visit Google AI Studio:</span>{' '}
                <a
                  href="https://makersuite.google.com/app/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 underline hover:text-blue-700 transition-colors"
                >
                  https://makersuite.google.com/app/apikey
                </a>
              </div>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-slate-900 text-white rounded-full flex items-center justify-center text-sm font-bold">
                2
              </span>
              <div>
                <span className="font-semibold">Sign in</span> with your Google account
              </div>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-slate-900 text-white rounded-full flex items-center justify-center text-sm font-bold">
                3
              </span>
              <div>
                <span className="font-semibold">Create a new API key</span> or use an existing one
              </div>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-slate-900 text-white rounded-full flex items-center justify-center text-sm font-bold">
                4
              </span>
              <div>
                <span className="font-semibold">Copy the key</span> and paste it above
              </div>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 bg-slate-900 text-white rounded-full flex items-center justify-center text-sm font-bold">
                5
              </span>
              <div>
                <span className="font-semibold">Click "Save & Validate"</span> to test and save your key
              </div>
            </li>
          </ol>

          <div className="mt-6 p-4 bg-yellow-50/80 border-l-4 border-yellow-500 rounded-lg backdrop-blur-sm">
            <p className="text-sm text-yellow-800">
              <span className="font-semibold">🔒 Privacy Note:</span> Your API key is stored only in your browser's local storage and is never sent to any server except Google's Gemini API.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Export function to get the API key
export const getApiKey = (): string | null => {
  return localStorage.getItem(API_KEY_STORAGE_KEY);
};
