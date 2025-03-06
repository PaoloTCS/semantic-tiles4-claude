# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/reset.py
#!/usr/bin/env python
"""
reset.py
Utility script to reset the Semantic Tiles application data.
"""

import os
import json
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reset_application_data():
    """Reset all application data by clearing uploads folder and resetting domains."""
    try:
        # Get the backend directory (the directory this script is in)
        backend_dir = Path(__file__).parent.absolute()
        uploads_dir = backend_dir / 'uploads'
        
        # Check if uploads directory exists
        if not uploads_dir.exists():
            logger.warning(f"Uploads directory not found at {uploads_dir}")
            logger.info("Creating uploads directory structure")
            os.makedirs(uploads_dir, exist_ok=True)
        
        # Clear documents directory
        documents_dir = uploads_dir / 'documents'
        if documents_dir.exists():
            logger.info(f"Removing all documents from {documents_dir}")
            shutil.rmtree(documents_dir)
        
        # Create documents directory
        os.makedirs(documents_dir, exist_ok=True)
        
        # Create/reset data directory
        data_dir = uploads_dir / 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        # Create empty domains.json
        domains_file = data_dir / 'domains.json'
        empty_domains = {
            "domains": {},
            "rootDomains": []
        }
        
        logger.info(f"Creating empty domains.json at {domains_file}")
        with open(domains_file, 'w') as f:
            json.dump(empty_domains, f, indent=2)
        
        # Clear cache directory
        cache_dir = uploads_dir / 'cache'
        if cache_dir.exists():
            logger.info(f"Removing all cached files from {cache_dir}")
            shutil.rmtree(cache_dir)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info("Application data has been reset successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error resetting application data: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting application data reset...")
    success = reset_application_data()
    
    if success:
        logger.info("Reset complete. The application now has fresh empty data.")
    else:
        logger.error("Reset failed. See logs for details.")