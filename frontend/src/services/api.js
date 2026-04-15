/**
 * API Client Service - Method-Agnostic Translation API
 *
 * This service works with ANY backend that implements the standard API contract:
 * - POST /api/set-context → Returns session_id, formality_level
 * - POST /api/translate → Returns translated_text, explanation, etc.
 */

import axios from 'axios';
import { getBackend } from '../config/backends';

/**
 * Set conversation context (Step 1 of translation flow)
 *
 * @param {string} relationship - Who the user is talking to
 * @param {string|null} situation - Optional situation description
 * @param {string} method - Which backend to use ('agent', 'papago_rules', 'ml_model')
 * @returns {Promise<Object>} Context response with session_id and formality level
 */
export const setContext = async (relationship, situation, method = 'agent') => {
  const backend = getBackend(method);

  try {
    const response = await axios.post(`${backend.baseURL}/api/set-context`, {
      relationship,
      situation: situation || null
    });

    return {
      success: true,
      data: response.data,
      method: method
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || error.message,
      method: method
    };
  }
};

/**
 * Translate text using session context (Step 2 of translation flow)
 *
 * @param {string} sessionId - Session ID from setContext
 * @param {string} text - English text to translate
 * @param {string} method - Which backend to use
 * @returns {Promise<Object>} Translation response
 */
export const translate = async (sessionId, text, method = 'agent') => {
  const backend = getBackend(method);

  try {
    const response = await axios.post(`${backend.baseURL}/api/translate`, {
      session_id: sessionId,
      text
    });

    return {
      success: true,
      data: response.data,
      method: method
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || error.message,
      method: method
    };
  }
};

/**
 * Check if a backend is available
 *
 * @param {string} method - Which backend to check
 * @returns {Promise<boolean>} Whether backend is reachable
 */
export const checkBackendHealth = async (method = 'agent') => {
  const backend = getBackend(method);

  try {
    const response = await axios.get(`${backend.baseURL}/health`, {
      timeout: 3000
    });
    return response.status === 200;
  } catch (error) {
    return false;
  }
};
