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

# Import torch for device detection
import torch

# Import sentence transformers for embedding generation
from sentence_transformers import SentenceTransformer

# Import NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Import FAISS for efficient similarity search
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.upload_folder = upload_folder
        self.cache_dir = cache_dir or os.path.join(upload_folder, 'cache')
        self.model_dir = os.path.join(upload_folder, 'models')
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
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
        
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load or initialize sentence transformer model
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
            except:
                logger.error("Failed to load even the fallback model")
                raise
        
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
    
    def load_state(self):
        """Load saved state from disk."""
        try:
            # Load user activities
            activities_path = os.path.join(self.cache_dir, 'user_activities.json')
            if os.path.exists(activities_path):
                try:
                    with open(activities_path, 'r') as f:
                        self.user_activities = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Corrupted user_activities.json file, creating new")
                    self.user_activities = []
            
            # Load user profile
            profile_path = os.path.join(self.cache_dir, 'user_profile.json')
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        self.user_profile = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Corrupted user_profile.json file, creating new")
                    self.user_profile = {'interests': {}}
            
            # Ensure proper structure
            if not isinstance(self.user_profile, dict):
                self.user_profile = {'interests': {}}
            if 'interests' not in self.user_profile:
                self.user_profile['interests'] = {}
                
        except Exception as e:
            logger.error(f"Error loading interest modeler state: {str(e)}")
            # Initialize with empty defaults
            self.user_activities = []
            self.user_profile = {'interests': {}}
    
    def save_state(self):
        """Save current state to disk."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Ensure user_activities is a serializable list
            if not isinstance(self.user_activities, list):
                logger.warning("user_activities is not a list, resetting to empty list")
                self.user_activities = []
            
            # Validate user_activities for JSON serializability
            clean_activities = []
            for activity in self.user_activities:
                try:
                    # Test json serialization
                    json.dumps(activity)
                    clean_activities.append(activity)
                except (TypeError, OverflowError):
                    logger.warning(f"Found non-serializable activity, skipping: {str(activity)[:100]}")
            
            # Save user activities
            activities_path = os.path.join(self.cache_dir, 'user_activities.json')
            with open(activities_path, 'w') as f:
                json.dump(clean_activities, f, indent=2)
            
            # Ensure user_profile is a serializable dict
            if not isinstance(self.user_profile, dict):
                logger.warning("user_profile is not a dict, resetting to empty dict")
                self.user_profile = {'interests': {}}
            
            # Create a clean copy for serialization
            profile_copy = {'interests': {}}
            if 'interests' in self.user_profile and isinstance(self.user_profile['interests'], dict):
                for k, v in self.user_profile['interests'].items():
                    if isinstance(v, dict) and 'score' in v:
                        try:
                            # Convert score to float to ensure serializability
                            score = float(v['score'])
                            profile_copy['interests'][k] = {
                                'score': score,
                                'count': int(v.get('count', 1)),
                                'last_updated': float(v.get('last_updated', time.time()))
                            }
                        except (TypeError, ValueError):
                            logger.warning(f"Invalid score value in interest {k}, skipping")
            
            # Save user profile
            profile_path = os.path.join(self.cache_dir, 'user_profile.json')
            with open(profile_path, 'w') as f:
                json.dump(profile_copy, f, indent=2)
                
        except PermissionError as pe:
            logger.error(f"Permission error saving state to {self.cache_dir}: {str(pe)}")
        except Exception as e:
            logger.error(f"Error saving interest modeler state: {str(e)}")
    
    def initialize_doc_index(self):
        """Initialize or load FAISS index for document similarity search."""
        index_path = os.path.join(self.cache_dir, 'faiss_index.bin')
        lookup_path = os.path.join(self.cache_dir, 'document_lookup.pkl')
        
        if os.path.exists(index_path) and os.path.exists(lookup_path):
            try:
                self.doc_index = faiss.read_index(index_path)
                with open(lookup_path, 'rb') as f:
                    self.doc_lookup = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                self.create_new_doc_index()
        else:
            self.create_new_doc_index()
    
    def create_new_doc_index(self):
        """Create a new FAISS index for document embeddings."""
        dimension = self.model.get_sentence_embedding_dimension()
        self.doc_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.doc_lookup = {}
    
    def save_doc_index(self):
        """Save the FAISS index and document lookup to disk."""
        index_path = os.path.join(self.cache_dir, 'faiss_index.bin')
        lookup_path = os.path.join(self.cache_dir, 'document_lookup.pkl')
        
        try:
            faiss.write_index(self.doc_index, index_path)
            with open(lookup_path, 'wb') as f:
                pickle.dump(self.doc_lookup, f)
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text with tokenization, stop word removal, and lemmatization.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        return " ".join(cleaned_tokens)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using Sentence Transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            preprocessed = self.preprocess_text(text)
            if not preprocessed:
                preprocessed = text  # Fallback to original text if preprocessing removes everything
            
            embedding = self.model.encode(preprocessed)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a random embedding as fallback
            return np.random.randn(self.model.get_sentence_embedding_dimension())
    
    def add_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Add a document to the interest model.
        
        Args:
            document_id: Unique document identifier
            text: Document text
            metadata: Additional document metadata
        """
        try:
            # Get embedding for the document
            embedding = self.get_embedding(text)
            
            # Normalize the embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Reshape for FAISS
            embedding_reshaped = embedding.reshape(1, -1).astype(np.float32)
            
            # Add to FAISS index
            self.doc_index.add(embedding_reshaped)
            
            # Store document lookup information
            idx = self.doc_index.ntotal - 1
            self.doc_lookup[idx] = {
                'id': document_id,
                'metadata': metadata or {},
                'keywords': self.extract_keywords(text, top_k=10),
                'timestamp': time.time()
            }
            
            # Save index periodically
            if self.doc_index.ntotal % 5 == 0:
                self.save_doc_index()
                
            # Track user activity with this document
            self.track_user_activity("document_added", text[:500], {
                'document_id': document_id,
                'document_name': metadata.get('name', 'Unnamed document') if metadata else 'Unnamed document'
            })
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
    
    def find_similar_documents(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to query text.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            if self.doc_index is None or self.doc_index.ntotal == 0:
                return []
            
            # Get embedding for query
            query_embedding = self.get_embedding(query_text)
            
            # Normalize the embedding
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Reshape for FAISS
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search for similar documents
            scores, indices = self.doc_index.search(query_embedding, min(top_k, self.doc_index.ntotal))
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx in self.doc_lookup:  # Valid index
                    doc_info = self.doc_lookup[idx]
                    results.append({
                        'id': doc_info['id'],
                        'score': float(score),
                        'metadata': doc_info['metadata'],
                        'keywords': doc_info.get('keywords', [])
                    })
            
            # Track this query in user activity
            self.track_user_activity("document_search", query_text, {
                'results_count': len(results)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Extract keywords from text using TF-IDF.
        
        Args:
            text: Text to analyze
            top_k: Number of keywords to extract
            
        Returns:
            List of keyword objects with scores
        """
        try:
            # Preprocess the text
            preprocessed = self.preprocess_text(text)
            if not preprocessed:
                return []
                
            # Create a TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_df=0.85, 
                min_df=2 if len(text) > 1000 else 1,  # Adjust min_df based on text length
                max_features=500,
                stop_words='english'
            )
            
            # Apply to the single document
            try:
                tfidf_matrix = vectorizer.fit_transform([preprocessed])
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top keywords
                dense = tfidf_matrix.todense()
                episode = dense[0].tolist()[0]
                scores = [(score, fnm) for fnm, score in zip(feature_names, episode)]
                scores.sort(reverse=True)
                
                # Return top k keywords with scores
                return [{'keyword': word, 'score': score} for score, word in scores[:top_k] if score > 0]
                
            except Exception as e:
                # Fallback to simple word frequency for very short texts
                words = preprocessed.split()
                freq = Counter(words)
                total = sum(freq.values())
                
                if total == 0:
                    return []
                
                return [
                    {'keyword': word, 'score': count / total} 
                    for word, count in freq.most_common(top_k)
                    if len(word) > 3  # Skip very short words
                ]
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def extract_interests(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Extract user interests from text.
        
        Args:
            text: Text to analyze
            top_k: Number of interests to extract
            
        Returns:
            List of interest categories with confidence scores
        """
        try:
            # Extract keywords as interests
            keywords = self.extract_keywords(text, top_k=top_k)
            
            # Track this activity
            self.track_user_activity("interest_extraction", text[:500], {
                'extracted_interests': keywords
            })
            
            return [{'category': kw['keyword'], 'score': kw['score']} for kw in keywords]
                
        except Exception as e:
            logger.error(f"Error extracting interests: {str(e)}")
            return []
    
    def track_user_activity(self, activity_type: str, content: str, metadata: Dict[str, Any] = None):
        """
        Track user activity for interest modeling.
        
        Args:
            activity_type: Type of activity (e.g., 'query', 'document_view')
            content: Content associated with the activity
            metadata: Additional metadata
        """
        try:
            activity = {
                'timestamp': time.time(),
                'type': activity_type,
                'content': content[:1000],  # Limit content length
                'metadata': metadata or {}
            }
            
            # Add to activity list
            self.user_activities.append(activity)
            
            # Limit the size of the activity list (keep last 100 activities)
            if len(self.user_activities) > 100:
                self.user_activities = self.user_activities[-100:]
            
            # Update user profile
            self.update_user_profile(activity)
            
            # Save state periodically
            if len(self.user_activities) % 5 == 0:
                self.save_state()
                
        except Exception as e:
            logger.error(f"Error tracking user activity: {str(e)}")
    
    def update_user_profile(self, activity: Dict[str, Any]):
        """
        Update user profile based on activity.
        
        Args:
            activity: User activity data
        """
        try:
            # Extract interests from the activity content
            interests = self.extract_keywords(activity['content'], top_k=3)
            
            # Initialize interests if not present
            if 'interests' not in self.user_profile:
                self.user_profile['interests'] = {}
            
            # Update interest scores
            for interest in interests:
                keyword = interest['keyword']
                score = interest['score']
                
                if keyword in self.user_profile['interests']:
                    # Exponential decay for existing interests
                    old_score = self.user_profile['interests'][keyword]['score']
                    old_count = self.user_profile['interests'][keyword]['count']
                    new_score = (old_score * old_count + score) / (old_count + 1)
                    self.user_profile['interests'][keyword] = {
                        'score': new_score,
                        'count': old_count + 1,
                        'last_updated': time.time()
                    }
                else:
                    # Add new interest
                    self.user_profile['interests'][keyword] = {
                        'score': score,
                        'count': 1,
                        'last_updated': time.time()
                    }
            
            # Age out old interests
            current_time = time.time()
            interests_to_remove = []
            
            for keyword, data in self.user_profile['interests'].items():
                # If interest hasn't been updated in 30 days, remove it
                if current_time - data['last_updated'] > 30 * 24 * 3600:
                    interests_to_remove.append(keyword)
                # Otherwise decay its score
                else:
                    days_old = (current_time - data['last_updated']) / (24 * 3600)
                    decay_factor = max(0.1, 1.0 - (days_old / 30))
                    data['score'] *= decay_factor
            
            # Remove aged out interests
            for keyword in interests_to_remove:
                del self.user_profile['interests'][keyword]
                
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
    
    def get_user_profile(self) -> Dict[str, Any]:
        """
        Get the current user interest profile.
        
        Returns:
            User profile data
        """
        try:
            # Ensure user_profile exists and is a dictionary
            if not isinstance(self.user_profile, dict):
                self.user_profile = {}
                
            # Get top interests sorted by score
            top_interests = []
            if 'interests' in self.user_profile and self.user_profile['interests']:
                try:
                    top_interests = sorted(
                        [{'category': k, 'score': float(v['score'])} for k, v in self.user_profile['interests'].items()
                         if isinstance(v, dict) and 'score' in v and v['score'] is not None],
                        key=lambda x: x['score'],
                        reverse=True
                    )[:10]  # Limit to top 10 interests
                except Exception as sort_error:
                    logger.error(f"Error sorting interests: {str(sort_error)}")
                    # Fall back to unsorted interests
                    top_interests = [{'category': k, 'score': v.get('score', 0.0) if isinstance(v, dict) else 0.0} 
                                   for k, v in self.user_profile['interests'].items()][:10]
            
            # Format the profile
            profile = {
                'top_interests': top_interests,
                'activity_count': len(getattr(self, 'user_activities', [])),
                'last_updated': time.time()
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {str(e)}")
            return {'top_interests': [], 'error': str(e)}
    
    def add_domain_activity(self, domain_id: str, text: str):
        """
        Track domain-related activity.
        
        Args:
            domain_id: Domain ID
            text: Domain text (name + description)
        """
        metadata = {'domain_id': domain_id}
        self.track_user_activity('domain_interaction', text, metadata)
    
    def add_query_activity(self, document_path: str, query: str):
        """
        Track document query activity.
        
        Args:
            document_path: Document path
            query: Query text
        """
        metadata = {'document_path': document_path}
        self.track_user_activity('document_query', query, metadata)
    
    def compute_domain_distances(self, domains: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        """
        Compute semantic distances between domains using embeddings.
        
        Args:
            domains: List of domains with name and optional description
            
        Returns:
            Dictionary mapping (domain1_id, domain2_id) to distance
        """
        try:
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
                    
                    # Also compute interest-weighted distance based on user profile
                    # (This could be used for personalized visualization)
                    # Right now, this is just storing the raw semantic distance
            
            return distances
            
        except Exception as e:
            logger.error(f"Error computing domain distances: {str(e)}")
            return {}