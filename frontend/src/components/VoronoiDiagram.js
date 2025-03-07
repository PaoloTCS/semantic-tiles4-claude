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
          const distanceKey = `${source}|${target}`;
          const reverseKey = `${target}|${source}`;
          
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
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        // Simple click to select AND navigate
        event.stopPropagation();
        
        // Navigate directly into the domain
        onDomainClick(d);
      });
      
    // Add tooltip for navigation hint
    domainNodes.append('title')
      .text(d => {
        const hasChildren = d.children && d.children.length > 0;
        return `${d.name}${hasChildren ? ' (click to view subdomains)' : ''}`;
      });
    
    // Add domain circles
    domainNodes.append('circle')
      .attr('r', d => Math.max(30, Math.min(50, 10 + d.documents?.length * 5 || 30)))
      .attr('fill', (d, i) => {
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        return color(i);
      })
      // Add special stroke for domains with children to indicate they can be navigated
      .attr('stroke', d => d.children && d.children.length > 0 ? '#FFD700' : '#fff')
      .attr('stroke-width', d => d.children && d.children.length > 0 ? 3 : 2);
    
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
    
    // Add delete button to each domain
    domainNodes.each(function(d) {
      const node = d3.select(this);
      
      // Add delete button (small circle with X)
      const deleteGroup = node.append('g')
        .attr('class', 'delete-button')
        .attr('transform', `translate(30, -30)`)
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
    
    // Show delete button on hover
    domainNodes.on('mouseover', function() {
      d3.select(this).select('.delete-button')
        .transition()
        .duration(200)
        .style('opacity', 1);
    })
    .on('mouseout', function() {
      d3.select(this).select('.delete-button')
        .transition()
        .duration(200)
        .style('opacity', 0);
    });
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      domainNodes.attr('transform', d => {
        // Keep nodes within bounds
        d.x = Math.max(60, Math.min(width - 60, d.x));
        d.y = Math.max(60, Math.min(height - 60, d.y));
        return `translate(${d.x},${d.y})`;
      });
    });
    
    // Clear selected domain on svg click
    svg.on('click', () => setSelectedDomain(null));
    
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