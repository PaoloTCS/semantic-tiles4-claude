// File: ~/VerbumTechnologies/semantic-tiles4-claude/frontend/src/App.js
import React, { useState, useEffect, useCallback } from 'react';
import BreadcrumbNav from './components/BreadcrumbNav';
import VoronoiDiagram from './components/VoronoiDiagram';
import DomainForm from './components/DomainForm';
import DocumentPanel from './components/DocumentPanel';
import DocumentUpload from './components/DocumentUpload';
import InterestVisualizer from './components/InterestVisualizer';
import ErrorBoundary from './components/ErrorBoundary';
import { 
  fetchDomains, 
  addDomain, 
  deleteDomain, 
  fetchDomainPath,
  trackUserActivity 
} from './services/apiService';
import './styles/App.css';

function App() {
  // State for domains at the current level
  const [domains, setDomains] = useState([]);
  
  // State for semantic distances
  const [semanticDistances, setSemanticDistances] = useState({});
  
  // Current parent ID (null for root level)
  const [currentParentId, setCurrentParentId] = useState(null);
  
  // Current domain details
  const [currentDomain, setCurrentDomain] = useState(null);
  
  // Selected domain for view (not navigation)
  const [selectedDomain, setSelectedDomain] = useState(null);
  
  // Breadcrumb path
  const [breadcrumbPath, setBreadcrumbPath] = useState([]);
  
  // Current document for document panel
  const [currentDocument, setCurrentDocument] = useState(null);
  
  // User activity tracking for interest modeling
  const [userActivity, setUserActivity] = useState('');
  
  // Loading state
  const [loading, setLoading] = useState(false);
  
  // Error state
  const [error, setError] = useState(null);
  
  // Success message state
  const [success, setSuccess] = useState(null);
  
  // Sidebar visible state
  const [sidebarVisible, setSidebarVisible] = useState(true);
  
  // Diagram dimensions
  const diagramWidth = 800;
  const diagramHeight = 600;
  
  // Helper: Track user activity
  const trackActivity = useCallback((type, content, metadata = {}) => {
    // Update local activity state
    setUserActivity(prev => {
      const combined = `${prev} ${content}`.trim();
      return combined.length > 1000 ? combined.substring(combined.length - 1000) : combined;
    });
    
    // Send to API for tracking
    trackUserActivity(type, content, metadata).catch(err => {
      console.error('Error tracking user activity:', err);
      // Non-critical, so just log the error
    });
  }, []);
  
  // Load domains from API (memoized to prevent unnecessary rerenders)
  const loadDomains = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchDomains(currentParentId);
      setDomains(data.domains || []);
      setSemanticDistances(data.semanticDistances || {});
      
      // If we have domain data, cache it in sessionStorage for resilience
      if (data.domains && data.domains.length > 0) {
        sessionStorage.setItem(`domains:${currentParentId || 'root'}`, JSON.stringify(data.domains));
        sessionStorage.setItem(`distances:${currentParentId || 'root'}`, JSON.stringify(data.semanticDistances || {}));
      }
    } catch (err) {
      console.error('Error loading domains:', err);
      
      // Try to load from cache if available
      const cachedDomains = sessionStorage.getItem(`domains:${currentParentId || 'root'}`);
      const cachedDistances = sessionStorage.getItem(`distances:${currentParentId || 'root'}`);
      
      if (cachedDomains) {
        setDomains(JSON.parse(cachedDomains));
        setSemanticDistances(cachedDistances ? JSON.parse(cachedDistances) : {});
        setError('Using cached data due to connection error. Some information may be outdated.');
      } else {
        // More specific error messages based on error type
        if (err.response && err.response.status === 401) {
          setError('Authentication error. Please check your API key.');
        } else if (err.code === 'ECONNABORTED' || !err.response) {
          setError('Network error. Please check your connection to the backend server.');
        } else {
          setError(`Failed to load domains: ${err.response?.data?.error || 'Unknown error'}`);
        }
      }
    } finally {
      setLoading(false);
    }
  }, [currentParentId]);
  
  // Load domain path for breadcrumbs
  const loadDomainPath = useCallback(async () => {
    if (!currentParentId) {
      setBreadcrumbPath([]);
      setCurrentDomain(null);
      return;
    }
    
    try {
      const path = await fetchDomainPath(currentParentId);
      setBreadcrumbPath(path);
      
      // Set current domain
      if (path.length > 0) {
        const currentPathDomain = path[path.length - 1];
        setCurrentDomain(currentPathDomain);
        
        // Track activity for current domain
        trackActivity(
          'domain_view', 
          `Domain: ${currentPathDomain.name} - ${currentPathDomain.description || ''}`,
          { domainId: currentPathDomain.id }
        );
        
        // Cache the path for resilience
        sessionStorage.setItem(`path:${currentParentId}`, JSON.stringify(path));
        
        // Display information about the current level in console for debugging
        console.log(`Current domain: ${currentPathDomain.name}`);
        console.log(`Children count: ${currentPathDomain.children?.length || 0}`);
        console.log(`Documents count: ${currentPathDomain.documents?.length || 0}`);
      }
    } catch (err) {
      console.error('Error loading domain path:', err);
      
      // Try to load from cache
      const cachedPath = sessionStorage.getItem(`path:${currentParentId}`);
      if (cachedPath) {
        const parsedPath = JSON.parse(cachedPath);
        setBreadcrumbPath(parsedPath);
        
        if (parsedPath.length > 0) {
          setCurrentDomain(parsedPath[parsedPath.length - 1]);
        }
      }
    }
  }, [currentParentId, trackActivity]);
  
  // Load domains at the current level
  useEffect(() => {
    loadDomains();
  }, [loadDomains]);
  
  // Load domain path when parent changes
  useEffect(() => {
    if (currentParentId) {
      loadDomainPath();
    } else {
      setBreadcrumbPath([]);
      setCurrentDomain(null);
      
      // Track activity for root level
      trackActivity('domain_view', 'Root level', { domainId: 'root' });
    }
  }, [currentParentId, loadDomainPath, trackActivity]);
  
  // Handle adding a new domain
  const handleAddDomain = async (name, description = '') => {
    try {
      await addDomain(name, currentParentId, description);
      
      // Track activity
      trackActivity(
        'domain_create', 
        `Created domain: ${name} - ${description}`,
        { domainName: name, parentId: currentParentId }
      );
      
      // Show success message
      setSuccess(`Successfully created ${currentParentId ? 'subdomain' : 'domain'}: ${name}`);
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
      
      // Refresh domains
      loadDomains();
    } catch (err) {
      console.error('Error adding domain:', err);
      setError('Failed to add domain. Please try again.');
    }
  };
  
  // Handle deleting a domain
  const handleDeleteDomain = async (domainId) => {
    try {
      setLoading(true); // Show loading state
      
      const domainToDelete = domains.find(d => d.id === domainId);
      const domainName = domainToDelete ? domainToDelete.name : domainId;
      
      await deleteDomain(domainId);
      
      // Track activity
      trackActivity(
        'domain_delete', 
        `Deleted domain: ${domainName}`,
        { domainId }
      );
      
      // Clear selected document if it belonged to the deleted domain
      if (currentDocument && domains.find(d => 
        d.documents && d.documents.some(doc => doc.id === currentDocument.id)
      )) {
        setCurrentDocument(null);
      }
      
      // Show success message temporarily
      setError(null);
      const successMessage = `Successfully deleted domain: ${domainName}`;
      setSuccess(successMessage);
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
      
      // Refresh domains
      loadDomains();
    } catch (err) {
      console.error('Error deleting domain:', err);
      setError(`Failed to delete domain: ${err.response?.data?.error || 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle domain click for drill-down
  const handleDomainClick = (domain) => {
    trackActivity(
      'domain_navigate', 
      `Navigated to domain: ${domain.name}`,
      { domainId: domain.id }
    );
    setCurrentParentId(domain.id);
    // Clear selected domain when navigating
    setSelectedDomain(null);
  };
  
  // Handle domain view (without navigation)
  const handleViewDomain = (domain) => {
    trackActivity(
      'domain_view', 
      `Viewed domain: ${domain.name}`,
      { domainId: domain.id }
    );
    setSelectedDomain(domain);
  };
  
  // Handle document click
  const handleDocumentClick = (document) => {
    trackActivity(
      'document_view', 
      `Viewed document: ${document.name}`,
      { documentId: document.id, documentPath: document.path }
    );
    setCurrentDocument(document);
  };
  
  // Handle breadcrumb navigation
  const handleBreadcrumbClick = (domainId) => {
    trackActivity(
      'breadcrumb_navigate', 
      `Navigated via breadcrumb${domainId ? ` to ${domainId}` : ' to root'}`,
      { domainId: domainId || 'root' }
    );
    setCurrentParentId(domainId);
  };
  
  // Handle closing the document panel
  const handleCloseDocumentPanel = () => {
    setCurrentDocument(null);
  };
  
  // Handle document upload completion
  const handleDocumentUpload = () => {
    trackActivity(
      'document_upload_complete', 
      `Document upload completed in domain ${currentDomain?.name || 'unknown'}`,
      { domainId: currentDomain?.id }
    );
    loadDomains();
  };
  
  // Toggle sidebar visibility
  const toggleSidebar = () => {
    setSidebarVisible(!sidebarVisible);
  };
  
  return (
    <ErrorBoundary>
      <div className="app-container">
        <header className="app-header">
          <div className="header-content">
            <h1 className="app-title">Semantic Tiles - Knowledge Map</h1>
            <button 
              className="sidebar-toggle"
              onClick={toggleSidebar}
              aria-label={sidebarVisible ? "Hide interests sidebar" : "Show interests sidebar"}
            >
              {sidebarVisible ? "⟩" : "⟨"}
            </button>
          </div>
        </header>
        
        <main className="main-content">
          <div className="primary-content">
            {/* Breadcrumb navigation */}
            <ErrorBoundary fallback={<div>Error loading navigation. <button onClick={() => setCurrentParentId(null)}>Return to root</button></div>}>
              <BreadcrumbNav 
                path={breadcrumbPath} 
                onNavigate={handleBreadcrumbClick} 
              />
            </ErrorBoundary>
            
            {/* Success message */}
            {success && (
              <div className="success-message">
                {success}
              </div>
            )}
            
            {/* Domain form */}
            <ErrorBoundary fallback={<div>Error in domain form component</div>}>
              <DomainForm 
                onAdd={handleAddDomain} 
                currentDomain={currentDomain}
                isSubdomain={currentParentId !== null}
              />
            </ErrorBoundary>
            
            {/* Document upload form (visible when inside a domain or when a domain is selected) */}
            {(currentDomain || selectedDomain) && (
              <ErrorBoundary fallback={<div>Error in document upload component</div>}>
                <DocumentUpload 
                  domainId={selectedDomain ? selectedDomain.id : currentDomain.id} 
                  onUploadComplete={handleDocumentUpload}
                />
              </ErrorBoundary>
            )}
            
            {/* Error message */}
            {error && (
              <div className="error-message">
                {error}
                {error.includes('connection') && 
                  <button 
                    className="retry-button" 
                    onClick={loadDomains}
                  >
                    Retry
                  </button>
                }
              </div>
            )}
            
            {/* Loading indicator */}
            {loading ? (
              <div className="loading-indicator">
                <div className="spinner"></div>
                <p>Loading domains...</p>
              </div>
            ) : (
              <ErrorBoundary fallback={<div>Error displaying knowledge map. <button onClick={loadDomains}>Reload</button></div>}>
                {/* Voronoi diagram */}
                {domains.length > 0 ? (
                  <VoronoiDiagram
                    domains={domains}
                    semanticDistances={semanticDistances}
                    width={diagramWidth}
                    height={diagramHeight}
                    onDomainClick={handleDomainClick}
                    onViewDomain={handleViewDomain}
                    onDocumentClick={handleDocumentClick}
                    onDeleteDomain={handleDeleteDomain}
                  />
                ) : (
                  <div className="empty-diagram">
                    <p>No domains at this level. Add your first domain!</p>
                  </div>
                )}
              </ErrorBoundary>
            )}
          </div>
          
          {/* Interest sidebar */}
          {sidebarVisible && (
            <aside className="sidebar">
              <ErrorBoundary fallback={<div>Error loading interest tracking</div>}>
                <InterestVisualizer
                  userActivity={userActivity}
                  width={300}
                  height={300}
                />
              </ErrorBoundary>
            </aside>
          )}
        </main>
        
        {/* Document panel */}
        <ErrorBoundary fallback={<div className="document-panel-error">Error displaying document</div>}>
          <DocumentPanel
            document={currentDocument}
            onClose={handleCloseDocumentPanel}
            onQuery={(query) => trackActivity('document_query', query, {documentId: currentDocument?.id})}
          />
        </ErrorBoundary>
      </div>
    </ErrorBoundary>
  );
}

export default App;