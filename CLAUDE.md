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

### Notes for Claude:

### Next steps for Claude:
~~1. Correct app so that when clicking on a created node, the interface will show that domain. For example from "Home", the start level, if one created domains Science and History, when clicking on Science, the app would show the created Science domain. At that level, one could add a subdomain, So, from Science, user could add physics and chemistry.~~ ✅ COMPLETED

~~In addition to the labeled nodes (Voronoi generators), I would also like to see the actual tessellation. At some point, we will have the "fuzzy Voronoi tessellation", but for now even basic tessellation is fine. The Voronoi/Delaunay tessellation dual graph allows us to also generate the edges between the generators. These edges have values. These values are important. Render these Delaunay edges to show their values.~~ ✅ COMPLETED

~~2. From any level below Home, one can add documents, e.g. PDFs, and have them displayed for AI extraction. So, from physics, user can add physicsExample.pdf, have it displayed, and then extract information from it (interrogate it with AI).~~ ✅ COMPLETED

### Implemented Features (March 2025):
1. **Domain Viewing and Navigation**:
   - Single click to view a domain without navigating into it
   - Domain details panel shows information and contained documents
   - Double-click to navigate into a domain (to see subdomains)
   - Navigation buttons appear on hover for easier interaction

2. **Voronoi Tessellation Visualization**:
   - Added Voronoi cell tessellation around domain nodes
   - Implemented Delaunay edges between connected domains
   - Edge weights (distances) are displayed on the diagram

3. **Document Management**:
   - Documents can be added to any domain or subdomain
   - Document upload UI is available when viewing a domain
   - Document contents can be viewed and queried

### Future Enhancements:
1. Add "fuzzy" Voronoi tessellation with overlapping regions
2. Enable AI to read documents and create domains/subdomains based on content
3. Improve domain visualization with better scaling and zooming capabilities

## Notes For re starts

Bash(cd /Users/paolopignatelli/VerbumTechnologies/semantic-tiles4-claude && echo "1. First run: python backend/reset.py …
      to reset the application data\n2. Then start the backend: python backend/run.py\n3. Finally start the frontend in a
      separate terminal: cd frontend && npm start")
  ⎿  1. First run: python backend/reset.py to reset the application data
     2. Then start the backend: python backend/run.py
     3. Finally start the frontend in a separate terminal: cd frontend && npm start