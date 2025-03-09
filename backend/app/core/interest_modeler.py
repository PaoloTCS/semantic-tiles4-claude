# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/app/core/interest_modeler.py
"""
app/core/interest_modeler.py
Models user interests based on browsing history and document interactions.
"""

import os
import numpy as np
import time
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import torch for device detection
import torch

# Import sentence transformers for embedding generation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available, using dummy embeddings")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import NLTK for text preprocessing
NLTK_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Initialize nltk data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
    NLTK_AVAILABLE = True
except ImportError:
    # Keep NLTK_AVAILABLE as False
    print("NLTK is not available, some NLP features will be limited")

# Import FAISS for efficient similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available, similarity search disabled")
    FAISS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using simple keyword extraction")
    SKLEARN_AVAILABLE = False 


class InterestModeler:
    """
    Models user interests based on document interactions and queries.
    Uses Sentence Transformers for embedding generation and FAISS for efficient similarity search.
    """
    
    def __init__(self, upload_folder: str, cache_dir: str = None):
        """
        Initialize the interest modeler with local models.
        
        Args:
            upload_folder: Path to uploaded files
            cache_dir: Directory to cache embeddings and models
        """
        global NLTK_AVAILABLE  # Add global declaration here
        global SENTENCE_TRANSFORMERS_AVAILABLE  # Add this line
        
        self.upload_folder = upload_folder
        self.cache_dir = cache_dir or os.path.join(upload_folder, 'cache')
        self.model_dir = os.path.join(upload_folder, 'models')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                # Download NLTK resources if not available
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('corpora/stopwords')
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('punkt')
                    nltk.download('stopwords')
                    nltk.download('wordnet')
                
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"NLTK initialization failed: {e}")
                NLTK_AVAILABLE = False  # Now properly modifies the global
        
        # Load or initialize sentence transformer model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                model_path = os.path.join(self.model_dir, 'all-MiniLM-L6-v2')
                if os.path.exists(model_path):
                    self.model = SentenceTransformer(model_path, device=self.device)
                else:
                    self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                    self.model.save(model_path)
            except Exception as e:
                logger.error(f"Error loading sentence transformer: {e}")
                # Fallback to a smaller model if available
                try:
                    self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device=self.device)
                except Exception as fallback_error:
                    logger.error(f"Failed to load fallback model: {fallback_error}")
                    SENTENCE_TRANSFORMERS_AVAILABLE = False
        
        # Initialize FAISS index for document similarity search
        self.doc_index = None
        self.doc_lookup = {}
        
        # Initialize user activity tracking
        self.user_activities = []
        self.user_profile = {}
        
        # Load saved state if available
        self.load_state()
        
        # Initialize FAISS index for documents
        self.initialize_doc_index()    


    # Add all the other methods, making sure to add "global NLTK_AVAILABLE" 
    # at the beginning of any method that modifies NLTK_AVAILABLE
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with tokenization, stop word removal, and lemmatization.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        global NLTK_AVAILABLE  # Add this for safety, even though this method doesn't modify it
        
        if not NLTK_AVAILABLE:
            # Simple fallback preprocessing
            return text.lower()
        
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stop words and lemmatize
            cleaned_tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token.isalnum() and token not in self.stop_words
            ]
            
            return " ".join(cleaned_tokens)
        except Exception as e:
            logger.error(f"Error in preprocess_text: {str(e)}")
            return text.lower()  # Fallback to simple lowercase
    
    def extract_interests(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Extract user interests from text.
        
        Args:
            text: Text to analyze
            top_k: Number of interests to extract
            
        Returns:
            List of interest categories with confidence scores
        """
        global NLTK_AVAILABLE  # Add this for consistent access
        
        try:
            # Extract keywords as interests
            keywords = self.extract_keywords(text, top_k=top_k)
            
            # Try to track this activity but continue if it fails
            try:
                self.track_user_activity("interest_extraction", text[:500], {
                    'extracted_interests': keywords
                })
            except Exception as tracking_error:
                logger.warning(f"Failed to track interest extraction: {tracking_error}")
                # Continue even if tracking fails
            
            return [{'category': kw['keyword'], 'score': kw['score']} for kw in keywords]
                
        except Exception as e:
            logger.error(f"Error extracting interests: {str(e)}")
            # Return some dummy interests as fallback
            return [
                {'category': f'interest{i}', 'score': (top_k-i)/top_k} 
                for i in range(min(3, top_k))
            ]        
        

    def compute_domain_distances(self, domains: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        """
        Compute semantic distances between domains using embeddings.
        
        Args:
            domains: List of domains with name and optional description
            
        Returns:
            Dictionary mapping (domain1_id, domain2_id) to distance
        """
        global NLTK_AVAILABLE  # Add for consistent access
        global SENTENCE_TRANSFORMERS_AVAILABLE
        
        try:
            # Check if we have the necessary dependencies
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("Cannot compute semantic distances: sentence-transformers not available")
                # Generate random distances as a fallback
                return self._generate_random_distances(domains)
            
            distances = {}
            
            # Get embeddings for each domain
            domain_embeddings = {}
            for domain in domains:
                text = domain['name']
                if 'description' in domain and domain['description']:
                    text += ": " + domain['description']
                    
                embedding = self.get_embedding(text)
                domain_embeddings[domain['id']] = embedding
            
            # Compute distances between all domain pairs
            for i, domain1_id in enumerate(domain_embeddings.keys()):
                domain1_embedding = domain_embeddings[domain1_id]
                for domain2_id in list(domain_embeddings.keys())[i+1:]:
                    domain2_embedding = domain_embeddings[domain2_id]
                    
                    # Compute cosine similarity
                    norm1 = np.linalg.norm(domain1_embedding)
                    norm2 = np.linalg.norm(domain2_embedding)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(domain1_embedding, domain2_embedding) / (norm1 * norm2)
                        # Convert to distance (0 to 1, where 0 is identical)
                        distance = 1 - max(-1.0, min(1.0, similarity))
                    else:
                        distance = 0.5  # Default distance for zero norm
                        
                    distances[(domain1_id, domain2_id)] = distance
            
            return distances
            
        except Exception as e:
            logger.error(f"Error computing domain distances: {str(e)}")
            return self._generate_random_distances(domains)