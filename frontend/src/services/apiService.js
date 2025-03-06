// File: ~/VerbumTechnologies/semantic-tiles4-claude/frontend/src/services/apiService.js
import axios from 'axios';

// Use an environment variable for the base URL in production
const baseURL = process.env.REACT_APP_BACKEND_URL
  ? `${process.env.REACT_APP_BACKEND_URL}/api`
  : '/api';

// Create an axios instance with timeout
const api = axios.create({
  baseURL,
  timeout: 30000, // 30 seconds timeout
});

/**
 * Fetch domains at a specific level
 * @param {string|null} parentId - Parent domain ID or null for root level
 * @returns {Promise<Object>} - Domains and semantic distances
 */
export const fetchDomains = async (parentId = null) => {
  try {
    const url = parentId 
      ? `/domains?parentId=${parentId}`
      : `/domains`;
    const response = await api.get(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching domains:', error);
    throw error;
  }
};

/**
 * Fetch a single domain by ID
 * @param {string} domainId - Domain ID
 * @returns {Promise<Object>} - Domain data
 */
export const fetchDomain = async (domainId) => {
  try {
    const response = await api.get(`/domains/${domainId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching domain:', error);
    throw error;
  }
};

/**
 * Fetch the path to a domain
 * @param {string} domainId - Domain ID
 * @returns {Promise<Array>} - Path to the domain
 */
export const fetchDomainPath = async (domainId) => {
  try {
    const response = await api.get(`/domains/${domainId}/path`);
    return response.data.path;
  } catch (error) {
    console.error('Error fetching domain path:', error);
    throw error;
  }
};

/**
 * Add a new domain
 * @param {string} name - Domain name
 * @param {string|null} parentId - Parent domain ID or null for root level
 * @param {string} description - Domain description
 * @returns {Promise<Object>} - New domain data
 */
export const addDomain = async (name, parentId = null, description = '') => {
  try {
    const response = await api.post(`/domains`, {
      name,
      parentId,
      description
    });
    return response.data;
  } catch (error) {
    console.error('Error adding domain:', error);
    throw error;
  }
};

/**
 * Update a domain
 * @param {string} domainId - Domain ID
 * @param {Object} updates - Updates to apply
 * @returns {Promise<Object>} - Updated domain data
 */
export const updateDomain = async (domainId, updates) => {
  try {
    const response = await api.put(`/domains/${domainId}`, updates);
    return response.data;
  } catch (error) {
    console.error('Error updating domain:', error);
    throw error;
  }
};

/**
 * Delete a domain
 * @param {string} domainId - Domain ID
 * @returns {Promise<Object>} - Success status
 */
export const deleteDomain = async (domainId) => {
  try {
    const response = await api.delete(`/domains/${domainId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting domain:', error);
    throw error;
  }
};

/**
 * Update domain positions
 * @param {Object} positions - Dictionary of domain_id -> {x, y}
 * @returns {Promise<Object>} - Success status
 */
export const updateDomainPositions = async (positions) => {
  try {
    const response = await api.post(`/domains/positions`, {
      positions
    });
    return response.data;
  } catch (error) {
    console.error('Error updating domain positions:', error);
    throw error;
  }
};

/**
 * Upload a document to a domain
 * @param {string} domainId - Domain ID
 * @param {File} file - Document file
 * @param {Function} onProgress - Progress callback
 * @returns {Promise<Object>} - Document data
 */
export const uploadDocument = async (domainId, file, onProgress = null) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domainId', domainId);
    
    const config = {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    };
    
    if (onProgress) {
      config.onUploadProgress = onProgress;
    }
    
    const response = await api.post(
      `/domains/${domainId}/documents`, 
      formData,
      config
    );
    return response.data;
  } catch (error) {
    console.error('Error uploading document:', error);
    throw error;
  }
};

/**
 * Remove a document from a domain
 * @param {string} domainId - Domain ID
 * @param {string} documentId - Document ID
 * @returns {Promise<Object>} - Success status
 */
export const removeDocument = async (domainId, documentId) => {
  try {
    const response = await api.delete(
      `/domains/${domainId}/documents/${documentId}`
    );
    return response.data;
  } catch (error) {
    console.error('Error removing document:', error);
    throw error;
  }
};

/**
 * Get document content by ID
 * @param {string} documentId - Document ID
 * @returns {Promise<Object>} - Document content and summary
 */
export const getDocumentContent = async (documentId) => {
  try {
    const response = await api.get(`/documents/${documentId}/content`);
    return response.data;
  } catch (error) {
    console.error('Error getting document content:', error);
    throw error;
  }
};

/**
 * Query a document
 * @param {string} documentId - Document ID
 * @param {string} query - Query text
 * @returns {Promise<Object>} - Query response
 */
export const queryDocument = async (documentId, query) => {
  try {
    const response = await api.post(
      `/documents/${documentId}/query`,
      { query }
    );
    return response.data;
  } catch (error) {
    console.error('Error querying document:', error);
    throw error;
  }
};

/**
 * Extract user interests from text
 * @param {string} text - Text to analyze
 * @returns {Promise<Object>} - Extracted interests
 */
export const extractInterests = async (text) => {
  try {
    const response = await api.post('/interests/extract', { text });
    return response.data;
  } catch (error) {
    console.error('Error extracting interests:', error);
    throw error;
  }
};

/**
 * Find documents similar to a query
 * @param {string} query - Query text
 * @param {number} topK - Number of results to return
 * @returns {Promise<Object>} - Similar documents
 */
export const findSimilarDocuments = async (query, topK = 5) => {
  try {
    const response = await api.post('/documents/similar', { 
      query,
      top_k: topK
    });
    return response.data;
  } catch (error) {
    console.error('Error finding similar documents:', error);
    throw error;
  }
};

/**
 * Get the current user interest profile
 * @returns {Promise<Object>} - User profile data
 */
export const getUserProfile = async () => {
  try {
    const response = await api.get('/interests/user-profile');
    return response.data;
  } catch (error) {
    console.error('Error getting user profile:', error);
    throw error;
  }
};

/**
 * Track user activity for interest modeling
 * @param {string} type - Activity type
 * @param {string} content - Activity content
 * @param {Object} metadata - Additional metadata
 * @returns {Promise<Object>} - Success status
 */
export const trackUserActivity = async (type, content, metadata = {}) => {
  try {
    const response = await api.post('/interests/track-activity', {
      type,
      content,
      metadata
    });
    return response.data;
  } catch (error) {
    console.error('Error tracking activity:', error);
    throw error;
  }
};