# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/app/core/domain_model.py
"""
app/core/domain_model.py
Models for knowledge domains and documents.
"""

import os
import json
import uuid
import logging
from typing import List, Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DomainStore:
    """
    Stores and manages knowledge domains and documents.
    Uses a JSON file-based storage system.
    """
    
    def __init__(self, storage_dir: str):
        """
        Initialize the domain store.
        
        Args:
            storage_dir: Directory for data storage
        """
        self.storage_dir = storage_dir
        self.data_file = os.path.join(storage_dir, 'domains.json')
        self.domains = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """
        Load domain data from storage.
        
        Returns:
            Domain data structure
        """
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            else:
                # Initialize with empty structure
                return {
                    "domains": {},
                    "rootDomains": []
                }
        except Exception as e:
            logger.error(f"Error loading domain data: {str(e)}")
            return {
                "domains": {},
                "rootDomains": []
            }
    
    def _save_data(self) -> bool:
        """
        Save domain data to storage.
        
        Returns:
            Success status
        """
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            with open(self.data_file, 'w') as f:
                json.dump(self.domains, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving domain data: {str(e)}")
            return False
    
    def get_domains(self, parent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get domains at a specific level.
        
        Args:
            parent_id: Parent domain ID or None for root level
            
        Returns:
            List of domain objects
        """
        try:
            domain_ids = []
            if parent_id is None:
                domain_ids = self.domains["rootDomains"]
            elif parent_id in self.domains["domains"]:
                domain_ids = self.domains["domains"][parent_id].get("children", [])
                
            return [
                self.domains["domains"][domain_id]
                for domain_id in domain_ids
                if domain_id in self.domains["domains"]
            ]
        except Exception as e:
            logger.error(f"Error getting domains: {str(e)}")
            return []
    
    def get_domain(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """
# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/app/core/domain_model.py
# (continuing from previous content)
        
        Args:
            domain_id: Domain ID
            
        Returns:
            Domain object or None if not found
        """
        try:
            if domain_id in self.domains["domains"]:
                return self.domains["domains"][domain_id]
            return None
        except Exception as e:
            logger.error(f"Error getting domain: {str(e)}")
            return None
    
    def add_domain(self, name: str, parent_id: Optional[str] = None, description: str = "") -> Optional[Dict[str, Any]]:
        """
        Add a new domain.
        
        Args:
            name: Domain name
            parent_id: Parent domain ID or None for root level
            description: Optional description
            
        Returns:
            New domain object or None on error
        """
        try:
            # Validate name
            if not name or not name.strip():
                logger.error("Cannot add domain with empty name")
                return None
                
            # Check for duplicate names at the same level
            existing_domains = self.get_domains(parent_id)
            if any(domain["name"].lower() == name.lower() for domain in existing_domains):
                logger.error(f"Domain with name '{name}' already exists at this level")
                return None
                
            # Check if parent exists if specified
            if parent_id is not None and parent_id not in self.domains["domains"]:
                logger.error(f"Parent domain with ID {parent_id} does not exist")
                return None
                
            # Create new domain
            domain_id = str(uuid.uuid4())
            domain = {
                "id": domain_id,
                "name": name,
                "description": description,
                "parentId": parent_id,
                "children": [],
                "documents": [],
                "x": 0,  # Initial position, will be set by frontend
                "y": 0
            }
            
            # Start a transaction-like operation
            original_domains = self.domains.copy()
            try:
                # Add to domains dictionary
                self.domains["domains"][domain_id] = domain
                
                # Add to parent's children or root domains
                if parent_id is None:
                    self.domains["rootDomains"].append(domain_id)
                else:
                    if "children" not in self.domains["domains"][parent_id]:
                        self.domains["domains"][parent_id]["children"] = []
                    self.domains["domains"][parent_id]["children"].append(domain_id)
                
                # Save changes
                if not self._save_data():
                    raise Exception("Failed to save domain data")
                    
                return domain
                
            except Exception as e:
                # Transaction failed - restore original data
                self.domains = original_domains
                logger.error(f"Transaction failed in add_domain: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error adding domain: {str(e)}")
            return None
    
    def update_domain(self, domain_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a domain.
        
        Args:
            domain_id: Domain ID
            updates: Dictionary of updates
            
        Returns:
            Updated domain object or None on error
        """
        try:
            if domain_id not in self.domains["domains"]:
                return None
                
            # Only update allowed fields
            allowed_fields = ['name', 'description', 'x', 'y']
            for field in allowed_fields:
                if field in updates:
                    self.domains["domains"][domain_id][field] = updates[field]
            
            self._save_data()
            return self.domains["domains"][domain_id]
        except Exception as e:
            logger.error(f"Error updating domain: {str(e)}")
            return None
    
    def delete_domain(self, domain_id: str) -> bool:
        """
        Delete a domain and its children.
        
        Args:
            domain_id: Domain ID
            
        Returns:
            Success status
        """
        try:
            if domain_id not in self.domains["domains"]:
                return False
                
            # Get domain info before deletion
            domain = self.domains["domains"][domain_id]
            parent_id = domain.get("parentId")
            
            # Recursively delete all children
            children = domain.get("children", [])
            for child_id in children:
                self.delete_domain(child_id)
            
            # Remove from parent's children list
            if parent_id is None:
                if domain_id in self.domains["rootDomains"]:
                    self.domains["rootDomains"].remove(domain_id)
            elif parent_id in self.domains["domains"]:
                if "children" in self.domains["domains"][parent_id]:
                    if domain_id in self.domains["domains"][parent_id]["children"]:
                        self.domains["domains"][parent_id]["children"].remove(domain_id)
            
            # Delete domain
            del self.domains["domains"][domain_id]
            
            self._save_data()
            return True
        except Exception as e:
            logger.error(f"Error deleting domain: {str(e)}")
            return False
    
    def add_document(self, domain_id: str, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a document to a domain.
        
        Args:
            domain_id: Domain ID
            document: Document object with name, path, etc.
            
        Returns:
            Updated document object or None on error
        """
        try:
            if domain_id not in self.domains["domains"]:
                return None
                
            document_id = document.get("id", str(uuid.uuid4()))
            document_obj = {
                "id": document_id,
                "name": document.get("name", "Untitled Document"),
                "path": document.get("path", ""),
                "type": document.get("type", "application/pdf"),
                "dateAdded": document.get("dateAdded", ""),
                "description": document.get("description", "")
            }
            
            if "documents" not in self.domains["domains"][domain_id]:
                self.domains["domains"][domain_id]["documents"] = []
                
            self.domains["domains"][domain_id]["documents"].append(document_obj)
            
            self._save_data()
            return document_obj
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return None
    
    def remove_document(self, domain_id: str, document_id: str) -> bool:
        """
        Remove a document from a domain.
        
        Args:
            domain_id: Domain ID
            document_id: Document ID
            
        Returns:
            Success status
        """
        try:
            if domain_id not in self.domains["domains"]:
                return False
                
            if "documents" not in self.domains["domains"][domain_id]:
                return False
                
            initial_count = len(self.domains["domains"][domain_id]["documents"])
            self.domains["domains"][domain_id]["documents"] = [
                doc for doc in self.domains["domains"][domain_id]["documents"]
                if doc.get("id") != document_id
            ]
            
            if len(self.domains["domains"][domain_id]["documents"]) < initial_count:
                self._save_data()
                return True
                
            return False
        except Exception as e:
            logger.error(f"Error removing document: {str(e)}")
            return False
    
    def get_domain_path(self, domain_id: str) -> List[Dict[str, Any]]:
        """
        Get the path from root to a domain.
        
        Args:
            domain_id: Domain ID
            
        Returns:
            List of domains in the path
        """
        try:
            path = []
            current_id = domain_id
            
            while current_id is not None:
                if current_id not in self.domains["domains"]:
                    break
                    
                domain = self.domains["domains"][current_id]
                path.insert(0, domain)
                current_id = domain.get("parentId")
                
            return path
        except Exception as e:
            logger.error(f"Error getting domain path: {str(e)}")
            return []
    
    def update_domain_positions(self, positions: Dict[str, Dict[str, float]]) -> bool:
        """
        Update domain positions.
        
        Args:
            positions: Dictionary of domain_id -> {x, y}
            
        Returns:
            Success status
        """
        try:
            for domain_id, position in positions.items():
                if domain_id in self.domains["domains"]:
                    if "x" in position:
                        self.domains["domains"][domain_id]["x"] = position["x"]
                    if "y" in position:
                        self.domains["domains"][domain_id]["y"] = position["y"]
            
            self._save_data()
            return True
        except Exception as e:
            logger.error(f"Error updating domain positions: {str(e)}")
            return False