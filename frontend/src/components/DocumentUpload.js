import React, { useState } from 'react';
import { uploadDocument } from '../services/apiService';

/**
 * DocumentUpload component for uploading documents to a domain
 */
const DocumentUpload = ({ domainId, onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };
  
  const handleUpload = async (e) => {
    e.preventDefault();
    
    if (!file || !domainId) return;
    
    setUploading(true);
    setProgress(0);
    setError(null);
    setSuccess(false);
    
    try {
      await uploadDocument(domainId, file, (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        setProgress(percentCompleted);
      });
      
      setSuccess(true);
      setFile(null);
      
      // Reset the file input
      const fileInput = document.getElementById('document-file');
      if (fileInput) {
        fileInput.value = '';
      }
      
      // Notify parent about upload completion
      if (onUploadComplete) {
        onUploadComplete();
      }
      
      // Reset success message after 3 seconds
      setTimeout(() => {
        setSuccess(false);
      }, 3000);
    } catch (err) {
      console.error('Error uploading document:', err);
      setError(err.response?.data?.error || 'Failed to upload document. Please try again.');
    } finally {
      setUploading(false);
    }
  };
  
  return (
    <div className="document-upload">
      <h3 className="upload-title">Upload Document</h3>
      
      <form onSubmit={handleUpload} className="upload-form">
        <div className="file-input-container">
          <input
            id="document-file"
            type="file"
            onChange={handleFileChange}
            accept=".pdf,.txt,.md,.doc,.docx"
            disabled={uploading}
          />
          {file && (
            <div className="selected-file">
              <span className="file-name">{file.name}</span>
              <span className="file-size">({Math.round(file.size / 1024)} KB)</span>
            </div>
          )}
        </div>
        
        {uploading && (
          <div className="upload-progress">
            <div 
              className="progress-bar"
              style={{ width: `${progress}%` }}
            ></div>
            <span className="progress-text">{progress}%</span>
          </div>
        )}
        
        {error && (
          <div className="upload-error">{error}</div>
        )}
        
        {success && (
          <div className="upload-success">Document uploaded successfully!</div>
        )}
        
        <button 
          type="submit" 
          className="upload-button"
          disabled={!file || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload Document'}
        </button>
      </form>
    </div>
  );
};

export default DocumentUpload;