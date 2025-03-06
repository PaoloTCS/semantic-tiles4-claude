# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/download_models.py
#!/usr/bin/env python
"""
download_models.py
Utility script to download required models for offline use.
"""

import os
import logging
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import torch
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_models():
    """Download and cache all required models."""
    try:
        # Create models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'uploads', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Device configuration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Download NLTK resources
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        # Download sentence transformer model
        logger.info("Downloading Sentence Transformer model...")
        st_model_path = os.path.join(models_dir, 'all-MiniLM-L6-v2')
        if not os.path.exists(st_model_path):
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            model.save(st_model_path)
        
        # Download Hugging Face models
        logger.info("Downloading embedding model...")
        hf_model_path = os.path.join(models_dir, 'sentence-transformers-all-MiniLM-L6-v2')
        if not os.path.exists(hf_model_path):
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            tokenizer.save_pretrained(hf_model_path)
            model.save_pretrained(hf_model_path)
        
        logger.info("Downloading summarization model...")
        summarization_model_path = os.path.join(models_dir, 'facebook-bart-large-cnn')
        if not os.path.exists(summarization_model_path):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)
            summarizer.save_pretrained(summarization_model_path)
        
        logger.info("Downloading QA model...")
        qa_model_path = os.path.join(models_dir, 'deepset-roberta-base-squad2')
        if not os.path.exists(qa_model_path):
            qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if device == "cuda" else -1)
            qa_model.save_pretrained(qa_model_path)
        
        logger.info("All models downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model download...")
    success = download_models()
    
    if success:
        logger.info("Model download complete. The application is ready to run offline.")
    else:
        logger.error("Model download failed. See logs for details.")