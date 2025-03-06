# File: ~/VerbumTechnologies/semantic-tiles4-claude/backend/app/__init__.py
"""
app/__init__.py
Application factory for Semantic Tiles.
"""

import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

def create_app(env):
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    # Allow requests from local development and the Netlify deployment
    CORS(app, resources={r"/*": {"origins": [
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://semantic-tiles4.netlify.app"
    ]}})
    
    # Configuration
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
        UPLOAD_FOLDER=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
        TEMPLATES_AUTO_RELOAD=True if env == 'development' else False
    )
    
    # Import and register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Default route
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'documents'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'data'), exist_ok=True)
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'cache'), exist_ok=True)
    
    return app