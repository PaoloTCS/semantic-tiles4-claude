# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/app/api/routes.py
"""
app/api/routes.py
API routes for the Semantic Tiles application.
"""

import os
import uuid
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app, send_file
from werkzeug.utils import secure_filename
from app.core.domain_model import DomainStore
from app.core.semantic_processor import SemanticProcessor
from app.core.interest_modeler import InterestModeler

# Create blueprint
api_bp = Blueprint('api', __name__)

# Setup global services
domain_store = None
semantic_processor = None
interest_modeler = None

def get_domain_store():
    """Get or initialize domain store."""
    global domain_store
    if domain_store is None:
        storage_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'data')
        os.makedirs(storage_dir, exist_ok=True)
        domain_store = DomainStore(storage_dir)
    return domain_store

def get_semantic_processor():
    """Get or initialize semantic processor."""
    global semantic_processor
    if semantic_processor is None:
        semantic_processor = SemanticProcessor(current_app.config['UPLOAD_FOLDER'])
    return semantic_processor

def get_interest_modeler():
    """Get or initialize interest modeler."""
    global interest_modeler
    if interest_modeler is None:
        interest_modeler = InterestModeler(current_app.config['UPLOAD_FOLDER'])
    return interest_modeler

@api_bp.route('/domains', methods=['GET'])
def get_domains():
    """Get domains at a specific level."""
    try:
        parent_id = request.args.get('parentId')
        
        # Get domains
        domain_store = get_domain_store()
        domains = domain_store.get_domains(parent_id)
        
        # Compute semantic distances if needed
        if domains and len(domains) > 1:
            interest_modeler = get_interest_modeler()
            distances = interest_modeler.compute_domain_distances(domains)
            
            # Format distances for output
            formatted_distances = {
                f"{k[0]}|{k[1]}": v for k, v in distances.items()
            }
            
            return jsonify({
                "domains": domains,
                "semanticDistances": formatted_distances
            })
        
        return jsonify({"domains": domains, "semanticDistances": {}})
        
    except Exception as e:
        current_app.logger.error(f"Error getting domains: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>', methods=['GET'])
def get_domain(domain_id):
    """Get a single domain by ID."""
    try:
        domain_store = get_domain_store()
        domain = domain_store.get_domain(domain_id)
        
        if domain:
            return jsonify(domain)
        else:
            return jsonify({"error": "Domain not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error getting domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains', methods=['POST'])
def add_domain():
    """Add a new domain."""
    try:
        data = request.json
        name = data.get('name')
        parent_id = data.get('parentId')
        description = data.get('description', '')
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        domain_store = get_domain_store()
        domain = domain_store.add_domain(name, parent_id, description)
        
        if domain:
            # Add the domain to interest modeler for tracking
            interest_modeler = get_interest_modeler()
            text = f"{name}: {description}"
            try:
                interest_modeler.add_domain_activity(domain['id'], text)
            except Exception as e:
                current_app.logger.warning(f"Error tracking domain activity: {str(e)}")
                # Continue even if tracking fails
                
            return jsonify(domain)
        else:
            return jsonify({"error": "Failed to add domain"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error adding domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>', methods=['PUT'])
def update_domain(domain_id):
    """Update a domain."""
    try:
        data = request.json
        updates = {}
        
        # Only allow specific fields to be updated
        allowed_fields = ['name', 'description', 'x', 'y']
        for field in allowed_fields:
            if field in data:
                updates[field] = data[field]
        
        domain_store = get_domain_store()
        domain = domain_store.update_domain(domain_id, updates)
        
        if domain:
            return jsonify(domain)
        else:
            return jsonify({"error": "Domain not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error updating domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>', methods=['DELETE'])
def delete_domain(domain_id):
    """Delete a domain and its children."""
    try:
        domain_store = get_domain_store()
        success = domain_store.delete_domain(domain_id)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Domain not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error deleting domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>/path', methods=['GET'])
def get_domain_path(domain_id):
    """Get the path from root to a domain."""
    try:
        domain_store = get_domain_store()
        path = domain_store.get_domain_path(domain_id)
        
        return jsonify({"path": path})
            
    except Exception as e:
        current_app.logger.error(f"Error getting domain path: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>/documents', methods=['POST'])
def add_document(domain_id):
    """Upload and add a document to a domain."""
    try:
        description = request.form.get('description', '')
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not domain_id:
            return jsonify({"error": "Domain ID is required"}), 400
            
        # Check if domain exists
        domain_store = get_domain_store()
        domain = domain_store.get_domain(domain_id)
        if not domain:
            return jsonify({"error": f"Domain with ID {domain_id} not found"}), 404
            
        # Save the file
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Validate file type (optional)
        allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
        if file_ext.lower() not in allowed_extensions:
            return jsonify({"error": f"File type {file_ext} not supported. Please upload a PDF, DOC, DOCX, or TXT file."}), 400
        
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'documents', unique_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        file.save(file_path)
        
        saved_file_exists = os.path.exists(file_path)
        if not saved_file_exists:
            return jsonify({"error": "Failed to save file"}), 500
            
        # Add document to domain
        document = {
            "id": str(uuid.uuid4()),
            "name": filename,
            "path": os.path.join('documents', unique_filename),
            "type": file.content_type,
            "dateAdded": datetime.now().isoformat(),
            "description": description
        }
        
        result = domain_store.add_document(domain_id, document)
        
        if result:
            # Extract text and index the document for interest modeling
            try:
                semantic_processor = get_semantic_processor()
                text = semantic_processor._extract_text_from_pdf(file_path)
                if text:
                    interest_modeler = get_interest_modeler()
                    interest_modeler.add_document(
                        document["id"], 
                        text, 
                        {
                            "name": document["name"],
                            "path": document["path"],
                            "domainId": domain_id
                        }
                    )
            except Exception as e:
                current_app.logger.warning(f"Error indexing document for interest modeling: {str(e)}")
                # Continue even if indexing fails
            
            return jsonify(result)
        else:
            # Clean up file if document not added
            if saved_file_exists:
                os.remove(file_path)
            return jsonify({"error": "Failed to add document to domain"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error adding document: {str(e)}")
        # Clean up file if exception occurs
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            current_app.logger.error(f"Error cleaning up file after exception: {str(cleanup_error)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>/documents/<document_id>', methods=['DELETE'])
def remove_document(domain_id, document_id):
    """Remove a document from a domain."""
    try:
        domain_store = get_domain_store()
        
        # Get the document info first so we can delete the file
        domain = domain_store.get_domain(domain_id)
        document = None
        
        if domain and "documents" in domain:
            for doc in domain["documents"]:
                if doc.get("id") == document_id:
                    document = doc
                    break
        
        success = domain_store.remove_document(domain_id, document_id)
        
        if success and document:
            # Delete the file if it exists
            if "path" in document:
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], document["path"])
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Document not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error removing document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/<path:document_path>', methods=['GET'])
def get_document(document_path):
    """Serve a document file."""
    try:
        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], document_path)
        
        if not os.path.exists(full_path):
            return jsonify({"error": "Document not found"}), 404
        
        # Check if the file is a PDF and set the correct mimetype
        if document_path.lower().endswith('.pdf'):
            return send_file(
                full_path,
                as_attachment=False,
                download_name=os.path.basename(document_path),
                mimetype='application/pdf'  # Explicitly set PDF mimetype
            )
        
        # For other file types, let Flask guess the mimetype
        return send_file(
            full_path,
            as_attachment=False,
            download_name=os.path.basename(document_path)
        )
            
    except Exception as e:
        current_app.logger.error(f"Error serving document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/<path:document_path>/summary', methods=['GET'])
def get_document_summary(document_path):
    """Get a summary of a document."""
    try:
        semantic_processor = get_semantic_processor()
        summary = semantic_processor.get_document_summary(document_path)
        
        return jsonify({"summary": summary})
            
    except Exception as e:
        current_app.logger.error(f"Error getting document summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/<path:document_path>/query', methods=['POST'])
def query_document(document_path):
    """Process a query about a document."""
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Validate document path
        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], document_path)
        if not os.path.exists(full_path):
            return jsonify({"error": "Document not found"}), 404
            
        # Validate query length
        if len(query) > 1000:  # Set a reasonable limit
            return jsonify({"error": "Query too long (maximum 1000 characters)"}), 400
            
        semantic_processor = get_semantic_processor()
        response = semantic_processor.process_document_query(document_path, query)
        
        # Track the query for interest modeling
        try:
            interest_modeler = get_interest_modeler()
            interest_modeler.add_query_activity(document_path, query)
        except Exception as e:
            current_app.logger.warning(f"Error tracking query activity: {str(e)}")
            # Continue even if tracking fails
        
        return jsonify({"response": response})
            
    except Exception as e:
        current_app.logger.error(f"Error querying document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/positions', methods=['POST'])
def update_domain_positions():
    """Update domain positions."""
    try:
        data = request.json
        positions = data.get('positions', {})
        
        domain_store = get_domain_store()
        success = domain_store.update_domain_positions(positions)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to update positions"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error updating domain positions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/interests/extract', methods=['POST'])
def extract_interests():
    """Extract interests from text."""
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
            
        interest_modeler = get_interest_modeler()
        interests = interest_modeler.extract_interests(text)
        
        return jsonify({"interests": interests})
            
    except Exception as e:
        current_app.logger.error(f"Error extracting interests: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/similar', methods=['POST'])
def find_similar_documents():
    """Find documents similar to query text."""
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        interest_modeler = get_interest_modeler()
        similar_docs = interest_modeler.find_similar_documents(query, top_k)
        
        return jsonify({"similar_documents": similar_docs})
            
    except Exception as e:
        current_app.logger.error(f"Error finding similar documents: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/interests/user-profile', methods=['GET'])
def get_user_profile():
    """Get the current user interest profile."""
    try:            
        interest_modeler = get_interest_modeler()
        profile = interest_modeler.get_user_profile()
        
        return jsonify({"profile": profile})
            
    except Exception as e:
        current_app.logger.error(f"Error getting user profile: {str(e)}")
        # Return an empty profile instead of an error to make the app more resilient
        return jsonify({"profile": {"top_interests": [], "activity_count": 0, "last_updated": 0}}), 200

@api_bp.route('/interests/track-activity', methods=['POST'])
def track_user_activity():
    """Track user activity for interest modeling."""
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 200
            
        activity_type = data.get('type', 'unknown')
        content = data.get('content', '')
        metadata = data.get('metadata', {})
        
        # Validate and sanitize input
        activity_type = str(activity_type)[:50]  # Limit type length
        
        # Ensure content is a string and limit its length
        if not isinstance(content, str):
            content = str(content)
        content = content[:2000]  # Limit content length
        
        # Ensure metadata is a dict
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Only proceed if we have meaningful content
        if not activity_type or not content:
            return jsonify({"success": False, "message": "Activity type and content are required but will not error"}), 200
            
        interest_modeler = get_interest_modeler()
        try:
            interest_modeler.track_user_activity(activity_type, content, metadata)
        except Exception as tracking_error:
            current_app.logger.error(f"Error in interest_modeler.track_user_activity: {str(tracking_error)}")
            # Continue without erroring the request
        
        return jsonify({"success": True})
            
    except Exception as e:
        current_app.logger.error(f"Error tracking user activity: {str(e)}")
        # Return success anyway to not disrupt the frontend
        return jsonify({"success": False, "message": "Activity logging failed but will not error"}), 200