import React, { useState } from 'react';

/**
 * DomainForm component for creating new domain nodes
 */
const DomainForm = ({ onAdd }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (name.trim() === '') return;
    
    onAdd(name.trim(), description.trim());
    setName('');
    setDescription('');
    setIsExpanded(false);
  };
  
  return (
    <div className="domain-form">
      {isExpanded ? (
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="domain-name">Domain Name</label>
            <input
              id="domain-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter domain name..."
              required
              autoFocus
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="domain-description">Description (optional)</label>
            <textarea
              id="domain-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter domain description..."
              rows={3}
            />
          </div>
          
          <div className="form-actions">
            <button 
              type="button" 
              className="cancel-button"
              onClick={() => setIsExpanded(false)}
            >
              Cancel
            </button>
            <button 
              type="submit" 
              className="submit-button"
              disabled={name.trim() === ''}
            >
              Add Domain
            </button>
          </div>
        </form>
      ) : (
        <button 
          className="add-domain-button" 
          onClick={() => setIsExpanded(true)}
        >
          + Add Domain
        </button>
      )}
    </div>
  );
};

export default DomainForm;