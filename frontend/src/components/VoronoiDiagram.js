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
  onDeleteDomain
}) => {
  const svgRef = useRef(null);
  const [selectedDomain, setSelectedDomain] = useState(null);
  
  // Create the diagram whenever domains or semanticDistances change
  useEffect(() => {
    if (!svgRef.current || domains.length === 0) return;
    
    // Clear previous diagram
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    // Create a simple force-directed layout for now
    // In a real implementation, this would use the semantic distances
    const simulation = d3.forceSimulation(domains)
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(60));
    
    // Distance-based forces using semantic distances
    if (Object.keys(semanticDistances).length > 0) {
      const links = [];
      
      // Create links between domains based on semantic distances
      for (let i = 0; i < domains.length; i++) {
        for (let j = i + 1; j < domains.length; j++) {
          const source = domains[i].id;
          const target = domains[j].id;
          const distanceKey = `${source}-${target}`;
          const reverseKey = `${target}-${source}`;
          
          if (semanticDistances[distanceKey] || semanticDistances[reverseKey]) {
            const distance = semanticDistances[distanceKey] || semanticDistances[reverseKey];
            links.push({
              source: i,
              target: j,
              distance: distance
            });
          }
        }
      }
      
      // Add force based on semantic distances
      simulation.force('link', d3.forceLink(links)
        .id((d, i) => i)
        .distance(d => Math.max(100, 200 * (1 - d.distance))));
    }
    
    // Create the main group
    const g = svg.append('g');
    
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
      .on('click', (event, d) => {
        event.stopPropagation();
        if (selectedDomain === d.id) {
          setSelectedDomain(null);
        } else {
          setSelectedDomain(d.id);
        }
      })
      .on('dblclick', (event, d) => {
        event.stopPropagation();
        onDomainClick(d);
      });
    
    // Add domain circles
    domainNodes.append('circle')
      .attr('r', d => Math.max(30, Math.min(50, 10 + d.documents?.length * 5 || 30)))
      .attr('fill', (d, i) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        return color(i);
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);
    
    // Add domain labels
    domainNodes.append('text')
      .text(d => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', '#fff')
      .attr('font-weight', 'bold')
      .attr('font-size', '14px')
      .attr('pointer-events', 'none');
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      domainNodes.attr('transform', d => {
        // Keep nodes within bounds
        d.x = Math.max(60, Math.min(width - 60, d.x));
        d.y = Math.max(60, Math.min(height - 60, d.y));
        return `translate(${d.x},${d.y})`;
      });
    });
    
    // Handle selected domain - show documents
    svg.on('click', () => setSelectedDomain(null));
    
    // Show documents for selected domain
    if (selectedDomain) {
      const domain = domains.find(d => d.id === selectedDomain);
      
      if (domain && domain.documents && domain.documents.length > 0) {
        // Find the domain node position
        const domainNode = domains.find(d => d.id === selectedDomain);
        
        // Create document list
        const documentsGroup = g.append('g')
          .attr('class', 'documents-panel')
          .attr('transform', `translate(${domainNode.x + 80}, ${domainNode.y - 100})`);
        
        // Add panel background
        documentsGroup.append('rect')
          .attr('width', 200)
          .attr('height', Math.min(domain.documents.length * 30 + 40, 300))
          .attr('fill', '#f8f9fa')
          .attr('stroke', '#dee2e6')
          .attr('rx', 5)
          .attr('ry', 5);
        
        // Add title
        documentsGroup.append('text')
          .text('Documents')
          .attr('x', 10)
          .attr('y', 20)
          .attr('font-weight', 'bold');
        
        // Add delete domain button
        documentsGroup.append('text')
          .text(' Delete Domain')
          .attr('x', 10)
          .attr('y', Math.min(domain.documents.length * 30 + 35, 295))
          .attr('class', 'delete-domain-btn')
          .attr('fill', '#dc3545')
          .attr('cursor', 'pointer')
          .on('click', (event) => {
            event.stopPropagation();
            if (window.confirm(`Are you sure you want to delete the domain "${domain.name}"?`)) {
              onDeleteDomain(domain.id);
            }
          });
        
        // Add document list
        const documentItems = documentsGroup.selectAll('.document-item')
          .data(domain.documents.slice(0, 8)) // Limit to 8 docs for simplicity
          .enter()
          .append('g')
          .attr('class', 'document-item')
          .attr('transform', (d, i) => `translate(10, ${i * 30 + 40})`)
          .on('click', (event, d) => {
            event.stopPropagation();
            onDocumentClick(d);
          });
        
        // Add document icons
        documentItems.append('text')
          .text('=Ä')
          .attr('x', 0)
          .attr('y', 0)
          .attr('dy', '0.5em');
        
        // Add document names
        documentItems.append('text')
          .text(d => d.name.length > 20 ? d.name.substring(0, 18) + '...' : d.name)
          .attr('x', 25)
          .attr('y', 0)
          .attr('dy', '0.5em')
          .attr('cursor', 'pointer')
          .attr('fill', '#0366d6');
        
        // Show "more" if there are more documents
        if (domain.documents.length > 8) {
          documentsGroup.append('text')
            .text(`+ ${domain.documents.length - 8} more...`)
            .attr('x', 10)
            .attr('y', 8 * 30 + 40)
            .attr('dy', '0.5em')
            .attr('font-style', 'italic')
            .attr('fill', '#6c757d');
        }
      }
    }
    
    return () => {
      simulation.stop();
    };
  }, [domains, semanticDistances, width, height, selectedDomain, onDomainClick, onDocumentClick, onDeleteDomain]);
  
  return (
    <div className="voronoi-diagram">
      <svg 
        ref={svgRef} 
        width={width} 
        height={height}
        className="diagram-svg"
      ></svg>
    </div>
  );
};

export default VoronoiDiagram;