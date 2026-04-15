import { useState } from 'react';
import { getAvailableBackends } from './config/backends';
import { setContext, translate } from './services/api';
import FormalityBadge from './components/FormalityBadge';

// Relationship options
const RELATIONSHIPS = [
  { value: 'boss', label: 'Boss', description: 'Your superior at work' },
  { value: 'elder', label: 'Elder', description: 'Older respected person' },
  { value: 'customer', label: 'Customer', description: 'Client or customer' },
  { value: 'teacher', label: 'Teacher', description: 'Instructor or mentor' },
  { value: 'colleague', label: 'Colleague', description: 'Coworker or peer' },
  { value: 'acquaintance', label: 'Acquaintance', description: 'Someone you know casually' },
  { value: 'friend', label: 'Friend', description: 'Close friend of similar age' },
  { value: 'younger_friend', label: 'Younger Friend', description: 'Younger friend' },
  { value: 'younger_sibling', label: 'Younger Sibling', description: 'Younger sibling' },
  { value: 'child', label: 'Child', description: 'Child or young person' }
];

function App() {
  // State management
  const [selectedMethod, setSelectedMethod] = useState('agent');
  const [sessionId, setSessionId] = useState(null);
  const [relationship, setRelationship] = useState('');
  const [situation, setSituation] = useState('');
  const [formality, setFormality] = useState(null);
  const [formalityDescription, setFormalityDescription] = useState('');
  const [inputText, setInputText] = useState('');
  const [translations, setTranslations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [contextSet, setContextSet] = useState(false);

  const availableBackends = getAvailableBackends();

  // Handle context setup (Step 1)
  const handleSetContext = async () => {
    if (!relationship) {
      setError('Please select a relationship');
      return;
    }

    setLoading(true);
    setError(null);

    const result = await setContext(relationship, situation, selectedMethod);

    if (result.success) {
      setSessionId(result.data.session_id);
      setFormality(result.data.formality_level);
      setFormalityDescription(result.data.formality_description);
      setContextSet(true);
    } else {
      setError(result.error);
    }

    setLoading(false);
  };

  // Handle translation (Step 2)
  const handleTranslate = async () => {
    if (!inputText.trim()) {
      setError('Please enter text to translate');
      return;
    }

    setLoading(true);
    setError(null);

    const result = await translate(sessionId, inputText, selectedMethod);

    if (result.success) {
      setTranslations([...translations, {
        id: Date.now(),
        original: result.data.original_text,
        translated: result.data.translated_text,
        formality: result.data.formality_level,
        explanation: result.data.explanation,
        romanization: result.data.romanization
      }]);
      setInputText(''); // Clear input after successful translation
    } else {
      if (result.error.includes('Session not found')) {
        setError('Session expired. Please set context again.');
        setContextSet(false);
        setSessionId(null);
      } else {
        setError(result.error);
      }
    }

    setLoading(false);
  };

  // Reset context (start new session)
  const handleResetContext = () => {
    setSessionId(null);
    setRelationship('');
    setSituation('');
    setFormality(null);
    setContextSet(false);
    setTranslations([]);
    setInputText('');
    setError(null);
  };

  // Copy to clipboard
  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Noonchi Translator
          </h1>
          <p className="text-lg text-gray-600">
            English to Korean with Culturally Appropriate Formality
          </p>
        </div>

        {/* Method Selector */}
        <div className="card mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Translation Method
          </label>
          <select
            value={selectedMethod}
            onChange={(e) => {
              if (contextSet) {
                if (confirm('Changing methods will start a new session. Continue?')) {
                  setSelectedMethod(e.target.value);
                  handleResetContext();
                }
              } else {
                setSelectedMethod(e.target.value);
              }
            }}
            className="select-field"
          >
            {availableBackends.map(backend => (
              <option key={backend.id} value={backend.id}>
                {backend.name}
              </option>
            ))}
          </select>
          <p className="mt-2 text-sm text-gray-500">
            {availableBackends.find(b => b.id === selectedMethod)?.description}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Input */}
          <div className="space-y-6">
            {/* Context Setup */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                Step 1: Who are you talking to?
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Relationship *
                  </label>
                  <select
                    value={relationship}
                    onChange={(e) => setRelationship(e.target.value)}
                    disabled={contextSet}
                    className="select-field"
                  >
                    <option value="">Select relationship...</option>
                    {RELATIONSHIPS.map(rel => (
                      <option key={rel.value} value={rel.value}>
                        {rel.label} - {rel.description}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Situation (optional)
                  </label>
                  <input
                    type="text"
                    value={situation}
                    onChange={(e) => setSituation(e.target.value)}
                    disabled={contextSet}
                    placeholder="e.g., business meeting, casual lunch"
                    className="input-field"
                  />
                </div>

                {!contextSet ? (
                  <button
                    onClick={handleSetContext}
                    disabled={loading || !relationship}
                    className="btn-primary w-full"
                  >
                    {loading ? 'Setting Context...' : 'Set Context'}
                  </button>
                ) : (
                  <div className="space-y-3">
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <p className="text-sm font-medium text-green-800 mb-2">
                        ✓ Context Set!
                      </p>
                      <p className="text-sm text-green-700 mb-2">
                        {formalityDescription}
                      </p>
                      <FormalityBadge level={formality} />
                    </div>
                    <button
                      onClick={handleResetContext}
                      className="btn-secondary w-full"
                    >
                      Change Context
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Translation Input */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                Step 2: Enter text to translate
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    English Text
                  </label>
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    disabled={!contextSet || loading}
                    placeholder={contextSet ? "Enter English text here..." : "Set context first"}
                    rows="4"
                    maxLength="5000"
                    className="textarea-field"
                  />
                  <p className="mt-1 text-sm text-gray-500 text-right">
                    {inputText.length} / 5000
                  </p>
                </div>

                <button
                  onClick={handleTranslate}
                  disabled={!contextSet || !inputText.trim() || loading}
                  className="btn-primary w-full"
                >
                  {loading ? 'Translating...' : 'Translate'}
                </button>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm font-medium text-red-800">Error</p>
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div>
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">
                Translation Results
              </h2>

              {translations.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <p>No translations yet</p>
                  <p className="text-sm mt-2">
                    Set context and enter text to see results
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  {translations.map((trans, index) => (
                    <div
                      key={trans.id}
                      className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
                    >
                      {/* Original Text */}
                      <div className="mb-3">
                        <p className="text-xs font-medium text-gray-500 uppercase mb-1">
                          English
                        </p>
                        <p className="text-gray-700">{trans.original}</p>
                      </div>

                      {/* Korean Translation */}
                      <div className="mb-3">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-xs font-medium text-gray-500 uppercase">
                            Korean
                          </p>
                          <button
                            onClick={() => copyToClipboard(trans.translated)}
                            className="text-xs text-blue-600 hover:text-blue-800"
                          >
                            Copy
                          </button>
                        </div>
                        <p className="text-2xl font-medium text-gray-900 mb-2">
                          {trans.translated}
                        </p>
                        <FormalityBadge level={trans.formality} />
                      </div>

                      {/* Romanization */}
                      {trans.romanization && (
                        <div className="mb-3">
                          <p className="text-xs font-medium text-gray-500 uppercase mb-1">
                            Pronunciation
                          </p>
                          <p className="text-sm text-gray-600 font-mono">
                            {trans.romanization}
                          </p>
                        </div>
                      )}

                      {/* Explanation */}
                      {trans.explanation && (
                        <details className="mt-3">
                          <summary className="text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700">
                            Explanation
                          </summary>
                          <p className="text-sm text-gray-600 mt-2">
                            {trans.explanation}
                          </p>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
