// File: ~/VerbumTechnologies/semantic-tiles4-claude/frontend/src/components/InterestVisualizer.js
import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as d3 from 'd3';
import { extractInterests, getUserProfile, findSimilarDocuments } from '../services/apiService';
import '../styles/InterestVisualizer.css';

/**
 * InterestVisualizer component for displaying and tracking user interests
 */
const InterestVisualizer = ({ userActivity, width = 300, height = 300 }) => {
  const [interests, setInterests] = useState([]);
  const [userProfile, setUserProfile] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('profile'); // 'profile' or 'current'
  
  const svgRef = useRef(null);
  const activityTimeoutRef = useRef(null);
  
  // Load user profile on mount
  useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        setLoading(true);
        const data = await getUserProfile();
        setUserProfile(data.profile);
        setError(null);
      } catch (err) {
        console.error('Error fetching user profile:', err);
        setError('Could not load user profile');
      } finally {
        setLoading(false);
      }
    };
    
    fetchUserProfile();
    
    // Refresh the profile periodically
    const intervalId = setInterval(() => {
      fetchUserProfile();
    }, 60000); // Every minute
    
    return () => clearInterval(intervalId);
  }, []);
  
  // Process user activity with debounce
  useEffect(() => {
    if (!userActivity || userActivity.trim() === '') return;
    
    // Clear existing timeout
    if (activityTimeoutRef.current) {
      clearTimeout(activityTimeoutRef.current);
    }
    
    // Set a new timeout for debouncing
    activityTimeoutRef.current = setTimeout(async () => {
      setLoading(true);
      setError(null);
      
      try {
        const data = await extractInterests(userActivity);
        setInterests(data.interests || []);
      } catch (err) {
        console.error('Error extracting interests:', err);
        setError('Failed to extract interests');
      } finally {
        setLoading(false);
      }
    }, 1000);
    
    return () => {
      if (activityTimeoutRef.current) {
        clearTimeout(activityTimeoutRef.current);
      }
    };
  }, [userActivity]);
  
  // Get recommendations based on interests
  const getRecommendations = useCallback(async () => {
    try {
      const interestsToUse = activeTab === 'profile' 
        ? (userProfile?.top_interests || [])
        : interests;
        
      if (interestsToUse.length === 0) return;
      
      // Use the findSimilarDocuments API with the top interest as query
      const topInterest = interestsToUse[0]?.category || '';
      if (!topInterest) return;
      
      const results = await findSimilarDocuments(topInterest, 3);
      setRecommendations(results.similar_documents || []);
    } catch (err) {
      console.error('Error getting recommendations:', err);
    }
  }, [activeTab, userProfile, interests]);
  
  // Update recommendations when interests change
  useEffect(() => {
    getRecommendations();
  }, [interests, userProfile, activeTab, getRecommendations]);
  
  // Render interests chart
  useEffect(() => {
    const data = activeTab === 'profile' 
      ? (userProfile?.top_interests || [])
      : interests;
      
    if (!svgRef.current || data.length === 0) return;
    
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    
    const margin = { top: 40, right: 20, bottom: 70, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    
    // Create scales
    const xScale = d3.scaleBand()
      .domain(data.map(d => d.category))
      .range([0, innerWidth])
      .padding(0.2);
    
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => d.score) * 1.1 || 1])
      .range([innerHeight, 0]);
    
    // Create chart group
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Add x-axis
    g.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .attr('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.15em');
    
    // Add y-axis
    g.append('g')
      .attr('class', 'y-axis')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${(d * 100).toFixed(0)}%`));
    
    // Add title
    g.append('text')
      .attr('class', 'chart-title')
      .attr('x', innerWidth / 2)
      .attr('y', -15)
      .attr('text-anchor', 'middle')
      .text(activeTab === 'profile' ? 'User Interest Profile' : 'Current Activity Interests');
    
    // Add bars
    g.selectAll('.bar')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.category))
      .attr('y', d => yScale(d.score))
      .attr('width', xScale.bandwidth())
      .attr('height', d => innerHeight - yScale(d.score))
      .attr('fill', (d, i) => {
        const colorScale = d3.scaleSequential(d3.interpolateBlues)
          .domain([0, data.length - 1]);
        return colorScale(i);
      });
    
    // Add labels
    g.selectAll('.label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', d => xScale(d.category) + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.score) - 5)
      .attr('text-anchor', 'middle')
      .text(d => `${(d.score * 100).toFixed(0)}%`);
    
  }, [interests, userProfile, activeTab, width, height]);
  
  return (
    <div className="interest-visualizer">
      <div className="tabs">
        <button 
          className={`tab-button ${activeTab === 'profile' ? 'active' : ''}`}
          onClick={() => setActiveTab('profile')}
        >
          User Profile
        </button>
        <button 
          className={`tab-button ${activeTab === 'current' ? 'active' : ''}`}
          onClick={() => setActiveTab('current')}
        >
          Current Activity
        </button>
      </div>
      
      {loading ? (
        <div className="interest-loading">
          <div className="spinner"></div>
          <p>Analyzing interests...</p>
        </div>
      ) : error ? (
        <div className="interest-error">{error}</div>
      ) : (activeTab === 'profile' && (!userProfile?.top_interests || userProfile.top_interests.length === 0)) ? (
        <div className="interest-empty">Your interest profile is still being built. Keep interacting with content!</div>
      ) : (activeTab === 'current' && interests.length === 0) ? (
        <div className="interest-empty">No interests detected in current activity. Try viewing or searching documents.</div>
      ) : (
        <svg ref={svgRef} width={width} height={height}></svg>
      )}
      
      {recommendations.length > 0 && (
        <div className="recommendations-section">
          <h4 className="recommendations-title">Recommended Content</h4>
          <ul className="recommendations-list">
            {recommendations.map(rec => (
              <li key={rec.id} className="recommendation-item">
                <div className="recommendation-name">{rec.metadata.name}</div>
                <div className="recommendation-score">
                  {Math.round(rec.score * 100)}% match
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default InterestVisualizer;