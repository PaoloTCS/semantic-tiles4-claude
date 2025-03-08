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
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not available, using dummy embeddings")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import NLTK for text preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    logger.warning("NLTK not available, using simple text processing")
    NLTK_AVAILABLE = False

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
                NLTK_AVAILABLE = False
        
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
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, document similarity search disabled")
            self.doc_index = None
            self.doc_lookup = {}
            return
            
        try:
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
        except Exception as e:
            logger.error(f"Error initializing document index: {str(e)}")
            self.doc_index = None
            self.doc_lookup = {}
    
    def create_new_doc_index(self):
        """Create a new FAISS index for document embeddings."""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, cannot create document index")
            self.doc_index = None
            self.doc_lookup = {}
            return
            
        try:
            # Get dimension from model or use standard size
            if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self, 'model'):
                try:
                    dimension = self.model.get_sentence_embedding_dimension()
                except:
                    dimension = 768  # Standard BERT embedding size
            else:
                dimension = 768
                
            self.doc_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.doc_lookup = {}
        except Exception as e:
            logger.error(f"Error creating new document index: {str(e)}")
            self.doc_index = None
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
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using Sentence Transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # If sentence transformers is not available, return random embedding
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a standard size for embeddings
            return np.random.randn(768)  # Standard BERT embedding size
            
        try:
            preprocessed = self.preprocess_text(text)
            if not preprocessed:
                preprocessed = text  # Fallback to original text if preprocessing removes everything
            
            embedding = self.model.encode(preprocessed)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a random embedding as fallback
            dim = 768  # Default embedding size if model is not accessible
            if hasattr(self, 'model') and hasattr(self.model, 'get_sentence_embedding_dimension'):
                try:
                    dim = self.model.get_sentence_embedding_dimension()
                except:
                    pass
            return np.random.randn(dim)
    
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
            
            # If sklearn is not available, use simple word counting
            if not SKLEARN_AVAILABLE:
                return self._extract_keywords_simple(preprocessed, top_k)
                
            # Create a TF-IDF vectorizer
            try:
                vectorizer = TfidfVectorizer(
                    max_df=0.85, 
                    min_df=2 if len(text) > 1000 else 1,  # Adjust min_df based on text length
                    max_features=500,
                    stop_words='english'
                )
                
                # Apply to the single document
                tfidf_matrix = vectorizer.fit_transform([preprocessed])
                feature_names = vectorizer.get_feature_names_out()
                
                # Get top keywords
                dense = tfidf_matrix.todense()
                episode = dense[0].tolist()[0]
                scores = [(score, fnm) for fnm, score in zip(feature_names, episode)]
                scores.sort(reverse=True)
                
                # Return top k keywords with scores
                return [{'keyword': word, 'score': score} for score, word in scores[:top_k] if score > 0]
            except Exception as sklearn_error:
                logger.warning(f"TF-IDF extraction failed: {str(sklearn_error)}")
                # Fallback to simple word counting
                return self._extract_keywords_simple(preprocessed, top_k)
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            # Generate some dummy keywords as a fallback
            return [
                {'keyword': f'topic{i}', 'score': (top_k-i)/top_k} 
                for i in range(min(3, top_k))
            ]
            
    def _extract_keywords_simple(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """Simple keyword extraction based on word frequency."""
        try:
            # Split into words
            words = text.split()
            # Count frequencies
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
            logger.error(f"Error in simple keyword extraction: {str(e)}")
            # Return some dummy keywords
            return [
                {'keyword': f'fallback{i}', 'score': (top_k-i)/top_k} 
                for i in range(min(3, top_k))
            ]
    
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
    
    def track_user_activity(self, activity_type: str, content: str, metadata: Dict[str, Any] = None):
        """
        Track user activity for interest modeling.
        
        Args:
            activity_type: Type of activity (e.g., 'query', 'document_view')
            content: Content associated with the activity
            metadata: Additional metadata
        """
        try:
            # Create sanitized activity object
            safe_metadata = {}
            if metadata:
                # Convert all metadata values to strings to ensure they're serializable
                for k, v in metadata.items():
                    try:
                        if isinstance(v, (dict, list)):
                            # Try to convert complex objects to simpler representations
                            safe_metadata[k] = str(v)[:100]  # Limit length
                        else:
                            safe_metadata[k] = str(v)[:100]  # Limit length
                    except:
                        safe_metadata[k] = "error_converting"
            
            activity = {
                'timestamp': time.time(),
                'type': str(activity_type)[:50],  # Limit and ensure string
                'content': str(content)[:1000] if content else "",  # Limit content length
                'metadata': safe_metadata
            }
            
            # Add to activity list
            if not hasattr(self, 'user_activities') or self.user_activities is None:
                self.user_activities = []
            
            self.user_activities.append(activity)
            
            # Limit the size of the activity list (keep last 100 activities)
            if len(self.user_activities) > 100:
                self.user_activities = self.user_activities[-100:]
            
            # Try to update user profile but continue if it fails
            try:
                self.update_user_profile(activity)
            except Exception as profile_error:
                logger.warning(f"Failed to update user profile: {profile_error}")
                # Continue even if profile update fails
            
            # Try to save state periodically
            if len(self.user_activities) % 5 == 0:
                try:
                    self.save_state()
                except Exception as save_error:
                    logger.warning(f"Failed to save activity state: {save_error}")
                    # Continue even if saving fails
                
        except Exception as e:
            logger.error(f"Error tracking user activity: {str(e)}")
            # Continue execution - tracking is non-critical
    
    def update_user_profile(self, activity: Dict[str, Any]):
        """
        Update user profile based on activity.
        
        Args:
            activity: User activity data
        """
        try:
            if not activity or 'content' not in activity:
                logger.warning("Cannot update profile: Invalid activity data")
                return
                
            # Try to extract interests from the activity content
            try:
                interests = self.extract_keywords(activity['content'], top_k=3)
            except Exception as kw_error:
                logger.warning(f"Failed to extract keywords for profile: {kw_error}")
                # Generate simple fallback keywords
                interests = [{'keyword': f'topic{i}', 'score': 0.9 - (i*0.1)} for i in range(3)]
            
            # Initialize user_profile if not present or invalid
            if not hasattr(self, 'user_profile') or not isinstance(self.user_profile, dict):
                self.user_profile = {}
                
            # Initialize interests if not present
            if 'interests' not in self.user_profile:
                self.user_profile['interests'] = {}
            
            # Update interest scores
            current_time = time.time()
            for interest in interests:
                try:
                    keyword = str(interest['keyword'])
                    score = float(interest['score'])
                    
                    if keyword in self.user_profile['interests']:
                        # Exponential decay for existing interests
                        old_score = float(self.user_profile['interests'][keyword].get('score', 0.5))
                        old_count = int(self.user_profile['interests'][keyword].get('count', 1))
                        new_score = (old_score * old_count + score) / (old_count + 1)
                        self.user_profile['interests'][keyword] = {
                            'score': new_score,
                            'count': old_count + 1,
                            'last_updated': current_time
                        }
                    else:
                        # Add new interest
                        self.user_profile['interests'][keyword] = {
                            'score': score,
                            'count': 1,
                            'last_updated': current_time
                        }
                except Exception as interest_error:
                    logger.warning(f"Error updating individual interest: {interest_error}")
                    continue  # Skip this interest but continue with others
            
            # Age out old interests
            interests_to_remove = []
            
            for keyword, data in self.user_profile['interests'].items():
                try:
                    # If interest hasn't been updated in 30 days, remove it
                    if current_time - data.get('last_updated', 0) > 30 * 24 * 3600:
                        interests_to_remove.append(keyword)
                    # Otherwise decay its score
                    else:
                        days_old = max(0, (current_time - data.get('last_updated', current_time)) / (24 * 3600))
                        decay_factor = max(0.1, 1.0 - (days_old / 30))
                        data['score'] = float(data.get('score', 0.5)) * decay_factor
                except Exception as decay_error:
                    logger.warning(f"Error processing interest decay: {decay_error}")
                    # Skip problematic interests
                    interests_to_remove.append(keyword)
            
            # Remove aged out interests
            for keyword in interests_to_remove:
                try:
                    del self.user_profile['interests'][keyword]
                except:
                    pass  # Ignore errors when removing
                
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            # Continue execution - profile updates are non-critical
    
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
    
    def _generate_random_distances(self, domains: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
        """
        Generate random distances as a fallback when semantic processing fails.
        
        Args:
            domains: List of domains
            
        Returns:
            Dictionary mapping (domain1_id, domain2_id) to random distance
        """
        try:
            distances = {}
            domain_ids = [domain['id'] for domain in domains]
            
            for i, domain1_id in enumerate(domain_ids):
                for domain2_id in domain_ids[i+1:]:
                    # Generate a random distance between 0.3 and 0.7
                    # This avoids extremes (identical or completely unrelated)
                    distance = 0.3 + (np.random.random() * 0.4)
                    distances[(domain1_id, domain2_id)] = distance
                    
            return distances
            
        except Exception as e:
            logger.error(f"Error generating random distances: {str(e)}")
            return {}