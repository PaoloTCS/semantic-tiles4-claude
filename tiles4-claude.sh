#!/bin/bash
# Create directory structure for semantic-tiles4-claude

BASE_DIR=~/VerbumTechnologies/semantic-tiles4-claude
echo "Creating directory structure in $BASE_DIR..."

# Create main directories
mkdir -p $BASE_DIR/backend/app/api
mkdir -p $BASE_DIR/backend/app/core
mkdir -p $BASE_DIR/backend/uploads/{documents,data,cache}
mkdir -p $BASE_DIR/frontend/src/{components,services,styles}
mkdir -p $BASE_DIR/frontend/public

# Create necessary files (empty for now)
touch $BASE_DIR/backend/run.py
touch $BASE_DIR/backend/reset.py
touch $BASE_DIR/backend/app/__init__.py
touch $BASE_DIR/backend/app/api/__init__.py
touch $BASE_DIR/backend/app/api/routes.py
touch $BASE_DIR/backend/app/core/__init__.py
touch $BASE_DIR/backend/app/core/domain_model.py
touch $BASE_DIR/backend/app/core/semantic_processor.py
touch $BASE_DIR/backend/app/core/interest_modeler.py

touch $BASE_DIR/frontend/src/App.js
touch $BASE_DIR/frontend/src/index.js
touch $BASE_DIR/frontend/src/components/VoronoiDiagram.js
touch $BASE_DIR/frontend/src/components/BreadcrumbNav.js
touch $BASE_DIR/frontend/src/components/DocumentPanel.js
touch $BASE_DIR/frontend/src/components/DocumentUpload.js
touch $BASE_DIR/frontend/src/components/DomainForm.js
touch $BASE_DIR/frontend/src/components/ErrorBoundary.js
touch $BASE_DIR/frontend/src/components/InterestVisualizer.js
touch $BASE_DIR/frontend/src/services/apiService.js
touch $BASE_DIR/frontend/src/styles/App.css
touch $BASE_DIR/frontend/src/styles/InterestVisualizer.css

touch $BASE_DIR/requirements.txt
touch $BASE_DIR/README.md

echo "Directory structure created successfully!"