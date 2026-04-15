/**
 * Backend configuration for different translation methods
 *
 * When you implement new translation methods (Papago+Rules, Pure ML),
 * just update the `available` flag to true and the frontend will automatically
 * support it!
 */

export const BACKENDS = {
  agent: {
    id: 'agent',
    name: 'AI Agent (Claude)',
    baseURL: 'http://localhost:8000',
    available: true,
    description: 'Intelligent AI-powered translation with detailed explanations',
    features: [
      'Natural language understanding',
      'Contextual formality application',
      'Detailed explanations',
      'Romanization support'
    ]
  },
  papago_rules: {
    id: 'papago_rules',
    name: 'Papago + Rules',
    baseURL: 'http://localhost:8001',
    available: false, // Set to true when Method #2 is implemented
    description: 'Traditional NLP with rule-based formality transformation',
    features: [
      'Papago API translation',
      'Morphological analysis',
      'Rule-based conjugation',
      'Honorific vocabulary mapping'
    ]
  },
  ml_model: {
    id: 'ml_model',
    name: 'Pure ML Model',
    baseURL: 'http://localhost:8002',
    available: false, // Set to true when Method #3 is implemented
    description: 'Custom trained neural network model',
    features: [
      'End-to-end learned translation',
      'Formality-aware training',
      'Fast inference',
      'No external API calls'
    ]
  }
};

// Helper function to get available backends
export const getAvailableBackends = () => {
  return Object.values(BACKENDS).filter(backend => backend.available);
};

// Helper function to get backend by ID
export const getBackend = (id) => {
  return BACKENDS[id] || BACKENDS.agent; // Default to agent
};
