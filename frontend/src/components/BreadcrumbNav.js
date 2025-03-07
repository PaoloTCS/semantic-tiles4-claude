import React from 'react';

/**
 * BreadcrumbNav component for displaying the current location within the domain hierarchy
 */
const BreadcrumbNav = ({ path = [], onNavigate }) => {
  return (
    <nav className="breadcrumb-nav">
      <ul className="breadcrumb-list">
        <li className="breadcrumb-item">
          <button 
            className="breadcrumb-link" 
            onClick={() => onNavigate(null)}
          >
            Home
          </button>
        </li>
        
        {path.map((item, index) => (
          <li key={item.id} className="breadcrumb-item">
            <span className="breadcrumb-separator">/</span>
            <button 
              className="breadcrumb-link" 
              onClick={() => onNavigate(item.id)}
              // Highlight the current domain
              style={{
                fontWeight: index === path.length - 1 ? 'bold' : 'normal'
              }}
            >
              {item.name}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default BreadcrumbNav;