import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DocumentViewer = ({ documentId }) => {
  const [pdfUrl, setPdfUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        setLoading(true);
        // Key change: Request as blob
        const response = await axios.get(`/api/documents/${documentId}/content`, {
          responseType: 'blob'
        });
        
        // Create object URL from blob
        const blob = new Blob([response.data], { type: 'application/pdf' });
        const url = URL.createObjectURL(blob);
        setPdfUrl(url);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching document content:', err);
        setError('Failed to load document');
        setLoading(false);
      }
    };
    
    if (documentId) {
      fetchDocument();
    }
    
    return () => {
      if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    };
  }, [documentId]);
  
  if (loading) return <div className="pdf-loading">Loading PDF document...</div>;
  if (error) return <div className="pdf-error">{error}</div>;
  if (!pdfUrl) return <div>No PDF document available</div>;
  
  return (
    <div className="pdf-viewer">
      <iframe 
        src={pdfUrl} 
        width="100%" 
        height="600px" 
        title="PDF Viewer"
        style={{ border: 'none' }}
      />
    </div>
  );
};

export default DocumentViewer;