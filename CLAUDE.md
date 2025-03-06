# Semantic Tiles Project Guidelines

## Build and Run Commands
- **Backend**: `python backend/run.py`
- **Backend Reset**: `python backend/reset.py` 
- **Frontend Dev**: `cd frontend && npm start`
- **Frontend Build**: `cd frontend && npm run build`
- **Frontend Test**: `cd frontend && npm test`

## Code Style Guidelines

### Python (Backend)
- **Naming**: Classes use PascalCase, functions/variables use snake_case, constants use UPPER_CASE
- **Imports**: Group in order: standard library, third-party packages, local modules
- **Type Hints**: Use type annotations for function parameters and return values
- **Error Handling**: Use specific exception handling with try/except blocks and proper logging
- **Documentation**: Use docstrings for classes and functions

### JavaScript/React (Frontend)
- **Naming**: Components use PascalCase, functions/variables use camelCase
- **Components**: Use functional components with hooks
- **Event Handlers**: Prefix with "handle" or "on" (e.g., handleUpload, onSubmit)
- **Imports**: Order by: React/hooks, components, services, styles
- **Error Handling**: Use error boundaries for component-level error handling

The codebase follows a clean, modular architecture with separation of concerns between backend API, semantic processing, and frontend visualization components.