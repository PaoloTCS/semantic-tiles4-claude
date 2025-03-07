import React, { useState } from 'react';

/**
 * DomainForm component for creating new domain nodes
 */
const DomainForm = ({ onAdd, currentDomain, isSubdomain }) => {
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
          {/* Added header showing current parent domain if adding a subdomain */}
          <div className="form-header">
            {isSubdomain ? (
              <h3 className="subdomain-header">
                Add Subdomain to "{currentDomain?.name || 'Domain'}"
              </h3>
            ) : (
              <h3>Add Domain</h3>
            )}
          </div>
          
          <div className="form-group">
            <label htmlFor="domain-name">
              {isSubdomain ? 'Subdomain Name' : 'Domain Name'}
            </label>
            <input
              id="domain-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={isSubdomain ? 
                `Enter subdomain name for ${currentDomain?.name}...` : 
                "Enter domain name..."}
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
              {isSubdomain ? 'Add Subdomain' : 'Add Domain'}
            </button>
          </div>
        </form>
      ) : (
        <button 
          className="add-domain-button" 
          onClick={() => setIsExpanded(true)}
        >
          {isSubdomain ? `+ Add Subdomain to "${currentDomain?.name}"` : '+ Add Domain'}
        </button>
      )}
    </div>
  );
};

export default DomainForm;