import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

/**
 * VoronoiDiagram component for visualizing domains and their relationships
 */
const VoronoiDiagram = ({ 
  domains = [], 
  semanticDistances = {}, 
  width = 800, 
  height = 600, 
  onDomainClick, 
  onDocumentClick,
  onDeleteDomain,
  onViewDomain
}) => {
  const svgRef = useRef(null);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [highlightedDomainId, setHighlightedDomainId] = useState(null);
  
  // Create the diagram whenever domains or semanticDistances change
  useEffect(() => {
    if (!svgRef.current || domains.length === 0) return;
    
    // Clear previous diagram
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    // Create a simple force-directed layout for now
    const simulation = d3.forceSimulation(domains)
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(60));
    
    // Track all positions for Voronoi
    const positions = [];
    
    // Create links for distance-based forces
    const links = [];
    
    // Create links between domains based on semantic distances
    for (let i = 0; i < domains.length; i++) {
      for (let j = i + 1; j < domains.length; j++) {
        const source = domains[i].id;
        const target = domains[j].id;
        const distanceKey = `${source}|${target}`;
        const reverseKey = `${target}|${source}`;
        
        if (semanticDistances[distanceKey] || semanticDistances[reverseKey]) {
          const distance = semanticDistances[distanceKey] || semanticDistances[reverseKey];
          links.push({
            source: i,
            target: j,
            distance: distance,
            domainAId: domains[i].id,
            domainBId: domains[j].id
          });
        }
      }
    }
    
    // Add force based on semantic distances if we have any
    if (links.length > 0) {
      simulation.force('link', d3.forceLink(links)
        .id((d, i) => i)
        .distance(d => Math.max(100, 200 * (1 - d.distance))));
    }
      
    // Create the main group
    const g = svg.append('g');
    
    // Create a container for Delaunay edges
    const edgesGroup = g.append('g').attr('class', 'delaunay-edges');
    
    // Create a container for Voronoi cells
    const voronoiGroup = g.append('g').attr('class', 'voronoi-cells');
    
    // Add zoom behavior
    svg.call(d3.zoom()
      .extent([[0, 0], [width, height]])
      .scaleExtent([0.5, 5])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      }));
    
    // Create domain nodes
    const domainNodes = g.selectAll('.domain-node')
      .data(domains)
      .enter()
      .append('g')
      .attr('class', 'domain-node')
      .attr('cursor', 'pointer');
    
    // Add click for viewing domain details
    domainNodes.on('click', (event, d) => {
      // First stop event propagation
      event.stopPropagation();
      
      // Highlight this domain
      setHighlightedDomainId(d.id);
      
      // Select this domain for viewing
      setSelectedDomain(d);
      
      // Call the onViewDomain callback to show domain details
      if (onViewDomain) {
        onViewDomain(d);
      }
    });
      
    // Add double-click for navigation
    domainNodes.on('dblclick', (event, d) => {
      event.stopPropagation();
      // Navigate into the domain on double-click
      if (onDomainClick) {
        onDomainClick(d);
      }
    });
    
    // Add tooltip for navigation hint
    domainNodes.append('title')
      .text(d => {
        const hasChildren = d.children && d.children.length > 0;
        return `${d.name} (Click to view, Double-click to navigate${hasChildren ? ' to subdomains' : ''})`;
      });
    
    // Add domain circles
    domainNodes.append('circle')
      .attr('r', d => Math.max(30, Math.min(50, 10 + d.documents?.length * 5 || 30)))
      .attr('fill', (d, i) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        return color(i);
      })
      // Add special stroke for highlighted domain and domains with children
      .attr('stroke', d => {
        if (d.id === highlightedDomainId) return '#FF4500'; // Bright orange for highlighted
        return d.children && d.children.length > 0 ? '#FFD700' : '#fff';
      })
      // Make the stroke wider for the highlighted domain
      .attr('stroke-width', d => {
        if (d.id === highlightedDomainId) return 5; // Thicker for highlighted
        return d.children && d.children.length > 0 ? 3 : 2;
      });
    
    // Add domain labels
    domainNodes.append('text')
      .text(d => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', '#fff')
      .attr('font-weight', 'bold')
      .attr('font-size', '14px')
      .attr('pointer-events', 'none');
    
    // Add indicator for domains with children
    domainNodes.filter(d => d.children && d.children.length > 0)
      .append('text')
      .text('▼')
      .attr('text-anchor', 'middle')
      .attr('dy', -25)
      .attr('fill', '#FFD700')
      .attr('font-weight', 'bold')
      .attr('font-size', '12px')
      .attr('pointer-events', 'none');
    
    // Add a "navigate" button to each domain
    domainNodes.filter(d => d.children && d.children.length > 0)
      .each(function(d) {
        const node = d3.select(this);
        
        // Add navigate button (small circle with arrow)
        const navGroup = node.append('g')
          .attr('class', 'navigate-button')
          .attr('transform', `translate(30, 0)`)
          .style('opacity', 0)  // Hidden by default
          .on('click', (event) => {
            event.stopPropagation();
            if (onDomainClick) {
              onDomainClick(d);
            }
          });
          
        // Navigate button background
        navGroup.append('circle')
          .attr('r', 12)
          .attr('fill', '#007bff')
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);
          
        // Arrow symbol
        navGroup.append('text')
          .attr('text-anchor', 'middle')
          .attr('dy', 4)
          .attr('fill', '#fff')
          .attr('font-weight', 'bold')
          .attr('pointer-events', 'none')
          .text('→');
      });
    
    // Add delete button to each domain
    domainNodes.each(function(d) {
      const node = d3.select(this);
      
      // Add delete button (small circle with X)
      const deleteGroup = node.append('g')
        .attr('class', 'delete-button')
        .attr('transform', `translate(0, -30)`)
        .style('opacity', 0)  // Hidden by default
        .on('click', (event) => {
          event.stopPropagation();
          if (window.confirm(`Are you sure you want to delete the domain "${d.name}"?`)) {
            onDeleteDomain(d.id);
          }
        });
        
      // Delete button background
      deleteGroup.append('circle')
        .attr('r', 12)
        .attr('fill', '#dc3545')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5);
        
      // X symbol
      deleteGroup.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', 4)
        .attr('fill', '#fff')
        .attr('font-weight', 'bold')
        .attr('pointer-events', 'none')
        .text('×');
    });
    
    // Show buttons on hover
    domainNodes.on('mouseover', function() {
      d3.select(this).select('.delete-button')
        .transition()
        .duration(200)
        .style('opacity', 1);
        
      d3.select(this).select('.navigate-button')
        .transition()
        .duration(200)
        .style('opacity', 1);
    })
    .on('mouseout', function() {
      d3.select(this).select('.delete-button')
        .transition()
        .duration(200)
        .style('opacity', 0);
        
      d3.select(this).select('.navigate-button')
        .transition()
        .duration(200)
        .style('opacity', 0);
    });
    
    // Update positions on simulation tick and create Voronoi diagram
    simulation.on('tick', () => {
      // Update node positions
      domainNodes.attr('transform', d => {
        // Keep nodes within bounds
        d.x = Math.max(60, Math.min(width - 60, d.x));
        d.y = Math.max(60, Math.min(height - 60, d.y));
        return `translate(${d.x},${d.y})`;
      });
      
      // Update positions array
      positions.length = 0;
      domains.forEach(d => {
        positions.push([d.x, d.y]);
      });
      
      // Create Voronoi diagram
      if (positions.length > 2) {
        // Clear previous cells
        voronoiGroup.selectAll('*').remove();
        
        // Create Delaunay triangulation
        const delaunay = d3.Delaunay.from(positions);
        
        // Create Voronoi diagram from Delaunay triangulation
        const voronoi = delaunay.voronoi([0, 0, width, height]);
        
        // Draw Voronoi cells
        voronoiGroup.selectAll('path')
          .data(domains)
          .enter()
          .append('path')
          .attr('d', (d, i) => voronoi.renderCell(i))
          .attr('fill', 'none')
          .attr('stroke', d => d.id === highlightedDomainId ? '#FF4500' : '#aaa')
          .attr('stroke-width', d => d.id === highlightedDomainId ? 3 : 1)
          .attr('opacity', d => d.id === highlightedDomainId ? 0.8 : 0.5);
          
        // Draw Delaunay edges with distances
        edgesGroup.selectAll('*').remove();
        
        // Draw edges between connected domains
        links.forEach(link => {
          if (typeof link.source === 'object' && typeof link.target === 'object') {
            const sourceX = link.source.x;
            const sourceY = link.source.y;
            const targetX = link.target.x;
            const targetY = link.target.y;
            const midX = (sourceX + targetX) / 2;
            const midY = (sourceY + targetY) / 2;
            
            // Draw line
            edgesGroup.append('line')
              .attr('x1', sourceX)
              .attr('y1', sourceY)
              .attr('x2', targetX)
              .attr('y2', targetY)
              .attr('stroke', '#555')
              .attr('stroke-width', 1)
              .attr('stroke-dasharray', '3,3')
              .attr('opacity', 0.6);
            
            // Draw distance value
            edgesGroup.append('text')
              .attr('x', midX)
              .attr('y', midY)
              .attr('text-anchor', 'middle')
              .attr('dy', -5)
              .attr('fill', '#333')
              .attr('font-size', '10px')
              .attr('opacity', 0.8)
              .text(link.distance.toFixed(2));
          }
        });
      }
    });
    
    // Clear selected domain on svg click
    svg.on('click', () => {
      setSelectedDomain(null);
    });
    
    return () => {
      simulation.stop();
    };
  }, [domains, semanticDistances, width, height, onDomainClick, onDocumentClick, onDeleteDomain, onViewDomain, highlightedDomainId]);
  
  return (
    <div className="voronoi-diagram">
      <svg 
        ref={svgRef} 
        width={width} 
        height={height}
        className="diagram-svg"
      ></svg>
      {selectedDomain && (
        <div className="domain-details">
          <h3>{selectedDomain.name}</h3>
          {selectedDomain.description && <p>{selectedDomain.description}</p>}
          {selectedDomain.children && selectedDomain.children.length > 0 && (
            <div className="domain-navigation">
              <button 
                className="navigate-button"
                onClick={() => onDomainClick && onDomainClick(selectedDomain)}
              >
                Navigate to {selectedDomain.name} ({selectedDomain.children.length} subdomains)
              </button>
            </div>
          )}
          {selectedDomain.documents && selectedDomain.documents.length > 0 && (
            <div className="domain-documents">
              <h4>Documents ({selectedDomain.documents.length})</h4>
              <ul>
                {selectedDomain.documents.map(doc => (
                  <li key={doc.id} onClick={() => onDocumentClick(doc)}>
                    {doc.name}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VoronoiDiagram;