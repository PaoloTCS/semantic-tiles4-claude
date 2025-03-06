# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/app/core/semantic_processor.py
"""
app/core/semantic_processor.py
Processes semantic information from documents using local models only.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from PyPDF2 import PdfReader
import torch
from transformers import pipeline, AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticProcessor:
    """
    Processes semantic information using local models only.
    Handles document processing, embedding generation, and semantic queries.
    """
    
    def __init__(self, upload_folder: str):
        """
        Initialize the semantic processor with local models.
        
        Args:
            upload_folder: Path to uploaded files
        """
        self.upload_folder = upload_folder
        self.embeddings_cache = {}
        
        # Initialize local models (load lazily to save memory)
        self._summarization_model = None
        self._qa_model = None
        self._embedding_model = None
        self._tokenizer = None
        
        # Device configuration 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(upload_folder, 'models')
        os.makedirs(self.model_path, exist_ok=True)
        
        logger.info(f"Semantic processor initialized with device: {self.device}")
    
    def get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            try:
                # First try to load from cache
                model_path = os.path.join(self.model_path, 'sentence-transformers-all-MiniLM-L6-v2')
                if os.path.exists(model_path):
                    self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                    self._embedding_model = AutoModel.from_pretrained(model_path).to(self.device)
                else:
                    # Download if not cached
                    self._tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                    self._embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
                    # Save to cache
                    self._tokenizer.save_pretrained(model_path)
                    self._embedding_model.save_pretrained(model_path)
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise
        return self._tokenizer, self._embedding_model
    
    def get_summarization_model(self):
        """Lazy load summarization model."""
        if self._summarization_model is None:
            logger.info("Loading summarization model...")
            try:
                model_name = "facebook/bart-large-cnn"  # Smaller model for summaries
                # Check if model exists in local cache
                model_path = os.path.join(self.model_path, model_name.replace('/', '-'))
                if os.path.exists(model_path):
                    self._summarization_model = pipeline("summarization", model=model_path, device=0 if self.device == "cuda" else -1)
                else:
                    self._summarization_model = pipeline("summarization", model=model_name, device=0 if self.device == "cuda" else -1)
                    # Save model to cache
                    self._summarization_model.save_pretrained(model_path)
            except Exception as e:
                logger.error(f"Error loading summarization model: {e}")
                # Fallback to a smaller model if the first one fails
                try:
                    self._summarization_model = pipeline("summarization", model="facebook/bart-base", device=0 if self.device == "cuda" else -1)
                except:
                    logger.error("Failed to load fallback model as well")
                    raise
        return self._summarization_model
    
    def get_qa_model(self):
        """Lazy load question answering model."""
        if self._qa_model is None:
            logger.info("Loading QA model...")
            try:
                model_name = "deepset/roberta-base-squad2"
                # Check if model exists in local cache
                model_path = os.path.join(self.model_path, model_name.replace('/', '-'))
                if os.path.exists(model_path):
                    self._qa_model = pipeline("question-answering", model=model_path, device=0 if self.device == "cuda" else -1)
                else:
                    self._qa_model = pipeline("question-answering", model=model_name, device=0 if self.device == "cuda" else -1)
                    # Save model to cache
                    self._qa_model.save_pretrained(model_path)
            except Exception as e:
                logger.error(f"Error loading QA model: {e}")
                raise
        return self._qa_model
    
    def compute_distances(self, items: List[Dict[str, Any]], level_id: Optional[str] = None) -> Dict[Tuple[str, str], float]:
        """
        Compute semantic distances between items.
        
        Args:
            items: List of items with name and optional description
            level_id: Optional ID of the current level for caching
            
        Returns:
            Dictionary mapping (item1_id, item2_id) to distance
        """
        try:
            distances = {}
            logger.info(f"Computing distances for {len(items)} items")
            
            # Get embeddings for each item
            item_embeddings = {}
            for item in items:
                embedding = None
                
                # If item has a document path, use that for embedding
                if 'documentPath' in item and item['documentPath']:
                    embedding = self._get_document_embedding(item['documentPath'])
                
                # Otherwise use the name and description
                if not embedding:
                    text = item['name']
                    if 'description' in item and item['description']:
                        text += ": " + item['description']
                    embedding = self._get_text_embedding(text)
                
                if embedding is not None:
                    item_embeddings[item['id']] = embedding
                else:
                    logger.warning(f"Could not generate embedding for item: {item['name']}")
                    # Create a random embedding as fallback to prevent crashes
                    item_embeddings[item['id']] = np.random.rand(768)  # BERT dimension
            
            # Compute distances between all item pairs
            for i, item1_id in enumerate(item_embeddings.keys()):
                item1_embedding = item_embeddings[item1_id]
                for item2_id in list(item_embeddings.keys())[i+1:]:
                    item2_embedding = item_embeddings[item2_id]
                    distance = self._compute_distance(item1_embedding, item2_embedding)
                    distances[(item1_id, item2_id)] = distance
            
            return distances
            
        except Exception as e:
            logger.error(f"Error computing distances: {str(e)}")
            return {}
    
    def get_document_summary(self, document_path: str) -> str:
        """
        Get a summary of a document using a local model.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Summary of the document
        """
        try:
            # Extract text from PDF
            full_path = os.path.join(self.upload_folder, document_path)
            text = self._extract_text_from_pdf(full_path)
            
            if not text:
                return "Could not extract text from document"
            
            # Get summarization model
            summarizer = self.get_summarization_model()
            
            # Split text into chunks if too large (BART models have ~1024 token limit)
            max_chunk_length = 1024
            chunks = [text[i:i+max_chunk_length] for i in range(0, min(len(text), 4096), max_chunk_length)]
            summaries = []
            
            for chunk in chunks:
                if len(chunk.strip()) < 50:  # Skip very small chunks
                    continue
                try:
                    result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
                    summaries.append(result[0]['summary_text'])
                except Exception as chunk_error:
                    logger.warning(f"Error summarizing chunk: {chunk_error}")
                    continue
            
            if not summaries:
                return f"Document preview: {text[:500]}..."
                
            return " ".join(summaries)
            
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            return f"Error summarizing document: {str(e)}"
    
    def process_document_query(self, document_path: str, query: str) -> str:
        """
        Process a query about a document using local QA model.
        
        Args:
            document_path: Path to the document
            query: User query about the document
            
        Returns:
            Response to the query
        """
        try:
            # Extract text from PDF
            full_path = os.path.join(self.upload_folder, document_path)
            text = self._extract_text_from_pdf(full_path)
            
            if not text:
                return "Could not extract text from document"
            
            # Get QA model
            qa_model = self.get_qa_model()
            
            # Process with QA pipeline - note: most QA models have context limits
            # So we need to find relevant sections first
            
            # Simple approach: split text into chunks and find most relevant chunk
            chunk_size = 512
            overlap = 50
            chunks = []
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk.strip()) > 100:  # Only keep meaningful chunks
                    chunks.append(chunk)
            
            if not chunks:
                return "The document doesn't contain enough text to answer questions."
            
            # Find most relevant chunks using simple keyword matching
            # (In a more advanced version, use embeddings similarity)
            query_words = set(query.lower().split())
            chunk_scores = []
            
            for chunk in chunks:
                words = set(chunk.lower().split())
                score = len(query_words.intersection(words))
                chunk_scores.append(score)
            
            # Get top chunks
            top_indices = sorted(range(len(chunk_scores)), key=lambda i: chunk_scores[i], reverse=True)[:3]
            context = " ".join([chunks[i] for i in top_indices])
            
            # Use QA model to get answer
            try:
                result = qa_model(question=query, context=context)
                answer = result['answer']
                
                # If confidence is low or answer is very short, provide more context
                if result['score'] < 0.1 or len(answer.split()) < 3:
                    return f"I'm not certain, but based on the document, I'd say: {answer}\n\nRelevant context: {context[:300]}..."
                
                return answer
            except Exception as qa_error:
                logger.error(f"QA model error: {qa_error}")
                return f"I couldn't find a specific answer to that question. Here's the most relevant section I found:\n\n{context[:300]}..."
                
        except Exception as e:
            logger.error(f"Error processing document query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def clear_cache(self):
        """Clear the embeddings cache."""
        self.embeddings_cache = {}
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            pdf = PdfReader(pdf_path)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _get_document_embedding(self, document_path: str) -> Optional[np.ndarray]:
        """
        Get embedding for a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Document embedding vector
        """
        try:
            cache_key = f"doc:{document_path}"
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            # Extract text from PDF
            full_path = os.path.join(self.upload_folder, document_path)
            text = self._extract_text_from_pdf(full_path)
            
            if not text:
                return None
            
            # Generate summary for embedding
            summary = f"Document: {os.path.basename(document_path)}\n\nContent: {text[:2000]}"
            embedding = self._get_text_embedding(summary)
            
            if embedding is not None:
                self.embeddings_cache[cache_key] = embedding
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting document embedding: {str(e)}")
            return None
    
    def _get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for text using local model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Cache handling
            cache_key = f"text:{hash(text)}"
            if cache_key in self.embeddings_cache:
                return self.embeddings_cache[cache_key]
            
            # Get models
            tokenizer, model = self.get_embedding_model()
            
            # Tokenize and encode
            encoded_input = tokenizer(text[:512], padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
                # Use mean pooling to get sentence embedding
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded_input['attention_mask']
                
                # Mask padding tokens
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                
                # Sum the masked embeddings
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.sum(input_mask_expanded, 1)
                
                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                
                # Get average embeddings
                embeddings = sum_embeddings / sum_mask
            
            # Convert to numpy and store in cache
            embedding = embeddings[0].cpu().numpy()
            self.embeddings_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting text embedding: {str(e)}")
            # Return random vector as fallback
            return np.random.randn(768)  # Typical embedding size for BERT-based models
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute semantic distance between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Distance value (0-1, where 0 is identical)
        """
        try:
            # Check if vectors are valid
            if emb1 is None or emb2 is None or len(emb1) == 0 or len(emb2) == 0:
                logger.warning("Invalid embedding vectors received for distance calculation")
                return 0.5  # Return moderate distance for invalid vectors
                
            # Check for NaN values 
            if np.isnan(emb1).any() or np.isnan(emb2).any():
                logger.warning("NaN values detected in embedding vectors")
                return 0.5
                
            # Normalize the vectors to avoid numerical issues
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                logger.warning("Zero norm encountered in embedding")
                return 0.5
                
            # Compute cosine similarity and convert to distance
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
            # Clip to valid range due to potential floating point errors
            similarity = max(-1.0, min(1.0, similarity))
            
            return max(0, min(1, 1 - similarity))  # Ensure distance is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error computing distance: {str(e)}")
            return 0.5  # Return moderate distance on error