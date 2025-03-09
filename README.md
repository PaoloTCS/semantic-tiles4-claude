# Semantic Tiles 4: Knowledge Map with Interest Tracking

A visual knowledge management system that organizes information using semantic relationships, Voronoi diagrams, and AI-powered interest modeling.

## Features

- Create hierarchical knowledge domains with semantic relationships
- Upload and manage documents within domains
- Visualize knowledge domains using Voronoi diagrams based on semantic similarity
- AI-powered interest tracking and modeling of user behavior
- Dynamic adaptation of visualizations based on user interests
- Query documents using semantic search
- Get document summaries with AI

## Technology Stack

- **Frontend**: React.js with D3.js for visualizations
- **Backend**: Flask REST API
- **ML/AI**: 
  - Sentence Transformers for semantic embeddings
  - FAISS for efficient vector similarity search
  - NLTK for text processing
  - OpenAI API for summaries and document queries

## Getting Started 

To recreate from Repo, and save the old local project: 
1. move OLD project to another folder

### Prerequisites

- Python 3.10+
- Node.js 16+
- OpenAI API key (optional, for document summaries and queries)

### Installation

1. Clone the repository


⏺ Here's how to recreate the full structure:

  1. Create Python virtual environment:
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

  2. Install backend dependencies:
  pip install -r requirements.txt

  3. Install frontend dependencies:
  cd frontend
  npm install

  4. Download required ML models:
  python backend/download_models.py

  5. Setup backend structure:
  mkdir -p backend/uploads/models
  mkdir -p backend/uploads/documents

 This succedssfully recreates the repo. 