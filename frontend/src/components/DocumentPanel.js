import React, { useState, useEffect } from 'react';
import { getDocumentContent, queryDocument } from '../services/apiService';

/**
 * DocumentPanel component for displaying document content and enabling Q&A
 */
const DocumentPanel = ({ document, onClose, onQuery }) => {
  const [content, setContent] = useState('');
  const [summary, setSummary] = useState('');
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('content'); // 'content', 'summary', or 'qa'
  
  // Load document content when document changes
  useEffect(() => {
    if (!document) return;
    
    const fetchContent = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const data = await getDocumentContent(document.id);
        setContent(data.content || '');
        setSummary(data.summary || '');
      } catch (err) {
        console.error('Error fetching document content:', err);
        setError('Failed to load document content. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchContent();
  }, [document]);
  
  // Reset state when document changes
  useEffect(() => {
    setQuery('');
    setAnswer(null);
    setTab('content');
  }, [document]);
  
  // Handle asking a question about the document
  const handleAskQuestion = async (e) => {
    e.preventDefault();
    
    if (!query.trim() || !document) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Track the query
      if (onQuery) {
        onQuery(query);
      }
      
      const data = await queryDocument(document.id, query);
      setAnswer(data.answer);
    } catch (err) {
      console.error('Error querying document:', err);
      setError('Failed to process your question. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // If no document is selected, don't render anything
  if (!document) return null;
  
  return (
    <div className={`document-panel ${document ? 'open' : ''}`}>
      <div className="document-panel-header">
        <div className="document-info">
          <h2 className="document-title">{document.name}</h2>
          {document.metadata && (
            <div className="document-metadata">
              {document.metadata.type && (
                <span className="document-type">{document.metadata.type}</span>
              )}
              {document.metadata.date && (
                <span className="document-date">{new Date(document.metadata.date).toLocaleDateString()}</span>
              )}
            </div>
          )}
        </div>
        <button 
          className="close-panel-button"
          onClick={onClose}
          aria-label="Close document panel"
        >
          
        </button>
      </div>
      
      <div className="document-panel-tabs">
        <button 
          className={`tab-button ${tab === 'content' ? 'active' : ''}`}
          onClick={() => setTab('content')}
        >
          Content
        </button>
        <button 
          className={`tab-button ${tab === 'summary' ? 'active' : ''}`}
          onClick={() => setTab('summary')}
        >
          Summary
        </button>
        <button 
          className={`tab-button ${tab === 'qa' ? 'active' : ''}`}
          onClick={() => setTab('qa')}
        >
          Ask Questions
        </button>
      </div>
      
      <div className="document-panel-content">
        {loading ? (
          <div className="document-loading">
            <div className="spinner"></div>
            <p>Loading...</p>
          </div>
        ) : error ? (
          <div className="document-error">{error}</div>
        ) : tab === 'content' ? (
          <div className="document-content">
            {content ? (
              <pre>{content}</pre>
            ) : (
              <p className="no-content">No content available for this document.</p>
            )}
          </div>
        ) : tab === 'summary' ? (
          <div className="document-summary">
            {summary ? (
              <p>{summary}</p>
            ) : (
              <p className="no-content">No summary available for this document.</p>
            )}
          </div>
        ) : (
          <div className="document-qa">
            <form onSubmit={handleAskQuestion} className="question-form">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question about this document..."
                className="question-input"
              />
              <button 
                type="submit" 
                className="ask-button"
                disabled={!query.trim() || loading}
              >
                Ask
              </button>
            </form>
            
            {answer && (
              <div className="answer-container">
                <h3 className="answer-label">Answer:</h3>
                <p className="answer-text">{answer}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentPanel;