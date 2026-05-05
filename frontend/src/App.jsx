import { useState } from 'react';
import { getAvailableBackends } from './config/backends';
import { setContext, translate } from './services/api';
import FormalityBadge from './components/FormalityBadge';

const RELATIONSHIPS = [
  { value: 'boss', label: 'Boss / Manager' },
  { value: 'professor', label: 'Professor / Teacher' },
  { value: 'elder', label: 'Elder (family or community)' },
  { value: 'colleague', label: 'Colleague' },
  { value: 'peer', label: 'Peer / Classmate' },
  { value: 'acquaintance', label: 'Acquaintance' },
  { value: 'stranger', label: 'Stranger' },
  { value: 'friend', label: 'Friend' },
  { value: 'subordinate', label: 'Subordinate / Junior' },
];

const SETTINGS = [
  { value: 'workplace', label: 'Workplace / Professional' },
  { value: 'academic', label: 'Academic / School' },
  { value: 'public', label: 'Public (restaurant, store, etc.)' },
  { value: 'social', label: 'Social gathering' },
  { value: 'intimate', label: 'Intimate / Home' },
];

function ageDiffLabel(val) {
  if (val === 0) return 'Similar age / unknown';
  return val > 0 ? `${val} years older than them` : `${Math.abs(val)} years younger than them`;
}

function ConfidencePip({ confidence }) {
  if (confidence == null) return null;
  const pct = Math.round(confidence * 100);
  const color = confidence >= 0.8 ? 'text-green-600' : confidence >= 0.6 ? 'text-yellow-600' : 'text-red-500';
  return (
    <span className={`text-xs font-medium ${color}`}>
      {pct}% confidence
    </span>
  );
}

function App() {
  const [selectedMethod, setSelectedMethod] = useState('agent');
  const [inputMode, setInputMode] = useState('text'); // 'text' | 'form'

  // Free-text path state
  const [situation, setSituation] = useState('');

  // Structured form state
  const [relationship, setRelationship] = useState('');
  const [ageDifferential, setAgeDifferential] = useState(0);
  const [setting, setSetting] = useState('');

  // Session / result state
  const [sessionId, setSessionId] = useState(null);
  const [formality, setFormality] = useState(null);
  const [formalityDescription, setFormalityDescription] = useState('');
  const [reasoning, setReasoning] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [detectedContext, setDetectedContext] = useState(null);

  // Translation state
  const [inputText, setInputText] = useState('');
  const [translations, setTranslations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [contextSet, setContextSet] = useState(false);

  const availableBackends = getAvailableBackends();

  const handleSetContext = async () => {
    let contextData;

    if (inputMode === 'text') {
      if (!situation.trim()) {
        setError('Please describe your situation');
        return;
      }
      contextData = { situation };
    } else {
      if (!relationship || !setting) {
        setError('Please select a relationship and setting');
        return;
      }
      contextData = { relationship, age_differential: ageDifferential, setting };
    }

    setLoading(true);
    setError(null);

    const result = await setContext(contextData, selectedMethod);

    if (result.success) {
      const d = result.data;
      setSessionId(d.session_id);
      setFormality(d.formality_token);
      setFormalityDescription(d.message);
      setReasoning(d.reasoning ?? null);
      setConfidence(d.confidence ?? null);
      setDetectedContext({
        relationship: d.relationship,
        ageDifferential: d.age_differential,
        setting: d.setting,
      });
      setContextSet(true);
    } else {
      setError(result.error);
    }

    setLoading(false);
  };

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
        formality: result.data.formality_token,
        explanation: result.data.explanation,
        romanization: result.data.romanization,
      }]);
      setInputText('');
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

  const handleResetContext = () => {
    setSessionId(null);
    setSituation('');
    setRelationship('');
    setAgeDifferential(0);
    setSetting('');
    setFormality(null);
    setFormalityDescription('');
    setReasoning(null);
    setConfidence(null);
    setDetectedContext(null);
    setContextSet(false);
    setTranslations([]);
    setInputText('');
    setError(null);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Noonchi Translator</h1>
          <p className="text-lg text-gray-600">English to Korean with Culturally Appropriate Formality</p>
        </div>

        {/* Method Selector */}
        <div className="card mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Translation Method</label>
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
              <option key={backend.id} value={backend.id}>{backend.name}</option>
            ))}
          </select>
          <p className="mt-2 text-sm text-gray-500">
            {availableBackends.find(b => b.id === selectedMethod)?.description}
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column */}
          <div className="space-y-6">

            {/* Step 1: Context */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Step 1: Who are you talking to?</h2>

              {/* Tabs */}
              {!contextSet && (
                <div className="flex border-b border-gray-200 mb-4">
                  <button
                    onClick={() => { setInputMode('text'); setError(null); }}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                      inputMode === 'text'
                        ? 'border-blue-600 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    Describe a situation
                  </button>
                  <button
                    onClick={() => { setInputMode('form'); setError(null); }}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                      inputMode === 'form'
                        ? 'border-blue-600 text-blue-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    Specify directly
                  </button>
                </div>
              )}

              <div className="space-y-4">
                {!contextSet ? (
                  <>
                    {inputMode === 'text' ? (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Situation
                        </label>
                        <textarea
                          value={situation}
                          onChange={(e) => setSituation(e.target.value)}
                          placeholder="e.g. 'Emailing my professor about a missed deadline' or 'Texting my friend to grab dinner'"
                          rows="3"
                          maxLength="500"
                          className="textarea-field"
                        />
                        <p className="mt-1 text-xs text-gray-400 text-right">{situation.length} / 500</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Relationship
                          </label>
                          <select
                            value={relationship}
                            onChange={(e) => setRelationship(e.target.value)}
                            className="select-field"
                          >
                            <option value="">Select relationship...</option>
                            {RELATIONSHIPS.map(r => (
                              <option key={r.value} value={r.value}>{r.label}</option>
                            ))}
                          </select>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Age differential — <span className="font-normal text-gray-500">{ageDiffLabel(ageDifferential)}</span>
                          </label>
                          <input
                            type="range"
                            min="-20"
                            max="20"
                            value={ageDifferential}
                            onChange={(e) => setAgeDifferential(parseInt(e.target.value))}
                            className="w-full accent-blue-600"
                          />
                          <div className="flex justify-between text-xs text-gray-400 mt-1">
                            <span>20 yrs younger</span>
                            <span>Similar age</span>
                            <span>20 yrs older</span>
                          </div>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            Setting
                          </label>
                          <select
                            value={setting}
                            onChange={(e) => setSetting(e.target.value)}
                            className="select-field"
                          >
                            <option value="">Select setting...</option>
                            {SETTINGS.map(s => (
                              <option key={s.value} value={s.value}>{s.label}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    )}

                    <button
                      onClick={handleSetContext}
                      disabled={loading || (inputMode === 'text' ? !situation.trim() : !relationship || !setting)}
                      className="btn-primary w-full"
                    >
                      {loading ? 'Setting Context...' : 'Set Context'}
                    </button>
                  </>
                ) : (
                  <div className="space-y-3">
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <p className="text-sm font-medium text-green-800 mb-2">✓ Context Set</p>

                      {/* Detected context fields */}
                      {detectedContext && (
                        <div className="mb-3 grid grid-cols-3 gap-2 text-xs text-gray-600">
                          <div>
                            <span className="font-medium text-gray-500 uppercase block">Relationship</span>
                            {detectedContext.relationship}
                          </div>
                          <div>
                            <span className="font-medium text-gray-500 uppercase block">Setting</span>
                            {detectedContext.setting}
                          </div>
                          <div>
                            <span className="font-medium text-gray-500 uppercase block">Age diff</span>
                            {detectedContext.ageDifferential === 0
                              ? 'similar'
                              : detectedContext.ageDifferential > 0
                                ? `+${detectedContext.ageDifferential}`
                                : detectedContext.ageDifferential}
                          </div>
                        </div>
                      )}

                      {/* Reasoning + confidence (free-text path only) */}
                      {reasoning && (
                        <div className="mb-2 flex items-start justify-between gap-2">
                          <p className="text-sm text-green-700 italic">{reasoning}</p>
                          <ConfidencePip confidence={confidence} />
                        </div>
                      )}

                      {confidence != null && confidence < 0.65 && (
                        <p className="text-xs text-yellow-700 mb-2">
                          Low confidence — if this looks wrong, try the "Specify directly" tab.
                        </p>
                      )}

                      <FormalityBadge level={formality} />
                    </div>
                    <button onClick={handleResetContext} className="btn-secondary w-full">
                      Change Context
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Step 2: Translate */}
            <div className="card">
              <h2 className="text-xl font-semibold mb-4">Step 2: Enter text to translate</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">English Text</label>
                  <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    disabled={!contextSet || loading}
                    placeholder={contextSet ? 'Enter English text here...' : 'Set context first'}
                    rows="4"
                    maxLength="5000"
                    className="textarea-field"
                  />
                  <p className="mt-1 text-sm text-gray-500 text-right">{inputText.length} / 5000</p>
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

            {/* Error */}
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
              <h2 className="text-xl font-semibold mb-4">Translation Results</h2>

              {translations.length === 0 ? (
                <div className="text-center py-12 text-gray-400">
                  <p>No translations yet</p>
                  <p className="text-sm mt-2">Set context and enter text to see results</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {translations.map((trans) => (
                    <div
                      key={trans.id}
                      className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 transition-colors"
                    >
                      <div className="mb-3">
                        <p className="text-xs font-medium text-gray-500 uppercase mb-1">English</p>
                        <p className="text-gray-700">{trans.original}</p>
                      </div>

                      <div className="mb-3">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-xs font-medium text-gray-500 uppercase">Korean</p>
                          <button
                            onClick={() => copyToClipboard(trans.translated)}
                            className="text-xs text-blue-600 hover:text-blue-800"
                          >
                            Copy
                          </button>
                        </div>
                        <p className="text-2xl font-medium text-gray-900 mb-2">{trans.translated}</p>
                        <FormalityBadge level={trans.formality} />
                      </div>

                      {trans.romanization && (
                        <div className="mb-3">
                          <p className="text-xs font-medium text-gray-500 uppercase mb-1">Pronunciation</p>
                          <p className="text-sm text-gray-600 font-mono">{trans.romanization}</p>
                        </div>
                      )}

                      {trans.explanation && (
                        <details className="mt-3">
                          <summary className="text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700">
                            Explanation
                          </summary>
                          <p className="text-sm text-gray-600 mt-2">{trans.explanation}</p>
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
