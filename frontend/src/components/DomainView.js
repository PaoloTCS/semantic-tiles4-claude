import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const DomainView = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [domain, setDomain] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [newSubdomainName, setNewSubdomainName] = useState('');
  const [newSubdomainDescription, setNewSubdomainDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    // Fetch domain details from API
    const fetchDomain = async () => {
      try {
        setLoading(true);
        const response = await fetch(`/api/domains/${id}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch domain (status ${response.status})`);
        }
        
        const domainData = await response.json();
        setDomain(domainData);
        setError(null);
      } catch (err) {
        console.error('Error fetching domain:', err);
        setError(`Failed to load domain: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    fetchDomain();
  }, [id]);

  const handleAddSubdomain = async (e) => {
    e.preventDefault();
    
    if (!newSubdomainName.trim()) {
      return;
    }
    
    try {
      setSubmitting(true);
      const response = await fetch(`/api/domains/${id}/subdomains`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newSubdomainName.trim(),
          description: newSubdomainDescription.trim() || undefined
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create subdomain (status ${response.status})`);
      }
      
      const newSubdomain = await response.json();
      
      // Update local state to include the new subdomain
      setDomain(prev => ({
        ...prev,
        children: [...(prev.children || []), newSubdomain]
      }));
      
      // Reset form
      setNewSubdomainName('');
      setNewSubdomainDescription('');
      
    } catch (err) {
      console.error('Error creating subdomain:', err);
      alert(`Failed to create subdomain: ${err.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const handleGoToSubdomain = (subdomainId) => {
    navigate(`/domain/${subdomainId}`);
  };

  const handleGoBack = () => {
    navigate(-1); // Go back to previous page
  };

  if (loading) {
    return <div className="loading">Loading domain information...</div>;
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={handleGoBack}>Go Back</button>
      </div>
    );
  }

  if (!domain) {
    return (
      <div className="not-found">
        <h2>Domain not found</h2>
        <button onClick={handleGoBack}>Go Back</button>
      </div>
    );
  }

  return (
    <div className="domain-view">
      <div className="domain-header">
        <button onClick={handleGoBack} className="back-button">
          &larr; Back
        </button>
        <h1>{domain.name}</h1>
      </div>

      {domain.description && (
        <div className="domain-description">
          <h3>Description</h3>
          <p>{domain.description}</p>
        </div>
      )}

      <div className="subdomains-section">
        <h2>Subdomains</h2>
        {domain.children && domain.children.length > 0 ? (
          <div className="subdomains-list">
            {domain.children.map(subdomain => (
              <div key={subdomain.id} className="subdomain-item">
                <h3>{subdomain.name}</h3>
                {subdomain.description && <p>{subdomain.description}</p>}
                <button 
                  onClick={() => handleGoToSubdomain(subdomain.id)}
                  className="view-button"
                >
                  View
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p>No subdomains yet. Add one below.</p>
        )}
      </div>

      <div className="add-subdomain-section">
        <h2>Add Subdomain</h2>
        <form onSubmit={handleAddSubdomain}>
          <div className="form-group">
            <label htmlFor="subdomain-name">Name:</label>
            <input
              id="subdomain-name"
              type="text"
              value={newSubdomainName}
              onChange={(e) => setNewSubdomainName(e.target.value)}
              placeholder="Enter subdomain name"
              required
              disabled={submitting}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="subdomain-description">Description (optional):</label>
            <textarea
              id="subdomain-description"
              value={newSubdomainDescription}
              onChange={(e) => setNewSubdomainDescription(e.target.value)}
              placeholder="Enter subdomain description"
              rows="3"
              disabled={submitting}
            ></textarea>
          </div>
          
          <button 
            type="submit" 
            className="submit-button"
            disabled={!newSubdomainName.trim() || submitting}
          >
            {submitting ? 'Creating...' : 'Add Subdomain'}
          </button>
        </form>
      </div>
      
      {domain.documents && domain.documents.length > 0 && (
        <div className="documents-section">
          <h2>Documents ({domain.documents.length})</h2>
          <ul className="documents-list">
            {domain.documents.map(doc => (
              <li key={doc.id} className="document-item">
                {doc.name}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DomainView;