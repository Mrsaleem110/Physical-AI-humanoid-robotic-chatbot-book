import React, { useState } from 'react';
import { useUser } from '../contexts/UserContext';

const ChapterPersonalization = ({ chapterId, content }) => {
  const { user } = useUser();
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [difficultyLevel, setDifficultyLevel] = useState('default');
  const [contentFormat, setContentFormat] = useState('default');

  const handlePersonalize = () => {
    if (!user) {
      alert('Please log in to personalize content');
      return;
    }

    setIsPersonalized(!isPersonalized);
  };

  const handleDifficultyChange = (e) => {
    setDifficultyLevel(e.target.value);
  };

  const handleFormatChange = (e) => {
    setContentFormat(e.target.value);
  };

  // Function to apply personalization to content based on user profile
  const personalizeContent = (content) => {
    if (!isPersonalized || !user) return content;

    let personalizedContent = content;

    // Apply personalization based on user's background
    if (user.softwareExperience === 'beginner' && difficultyLevel === 'default') {
      // Add more explanations for beginners
      personalizedContent = addBeginnerExplanations(content);
    } else if (user.softwareExperience === 'advanced' && difficultyLevel === 'default') {
      // Add more technical details for advanced users
      personalizedContent = addAdvancedDetails(content);
    }

    // Apply content format preferences
    if (contentFormat === 'visual') {
      // Add more diagrams and visual elements (in a real implementation)
      personalizedContent = addVisualElements(content);
    } else if (contentFormat === 'detailed') {
      // Add more detailed explanations
      personalizedContent = addDetailedExplanations(content);
    }

    return personalizedContent;
  };

  // Helper functions for personalization (simplified for this example)
  const addBeginnerExplanations = (content) => {
    // In a real implementation, this would add more basic explanations
    return `⚠️ [BEGINNER-FRIENDLY VERSION] ${content}`;
  };

  const addAdvancedDetails = (content) => {
    // In a real implementation, this would add more technical details
    return `🔬 [ADVANCED TECHNICAL DETAILS] ${content}`;
  };

  const addVisualElements = (content) => {
    // In a real implementation, this would add visual elements
    return `📊 [VISUAL ELEMENTS ADDED] ${content}`;
  };

  const addDetailedExplanations = (content) => {
    // In a real implementation, this would add more detailed explanations
    return `📝 [DETAILED EXPLANATIONS] ${content}`;
  };

  return (
    <div className="chapter-personalization" style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '12px',
      padding: '20px',
      margin: '20px 0',
      boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h4 style={{
          margin: '0',
          color: 'white',
          fontSize: '18px',
          fontWeight: '600'
        }}>
          🎯 Content Personalization
        </h4>
        <button
          onClick={handlePersonalize}
          style={{
            padding: '10px 16px',
            backgroundColor: isPersonalized ? '#28a745' : 'rgba(255,255,255,0.2)',
            color: isPersonalized ? 'white' : 'white',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            cursor: 'pointer',
            fontWeight: '600',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}
        >
          {isPersonalized ? 'Disable' : 'Enable'}
        </button>
      </div>

      {isPersonalized && user && (
        <div style={{
          marginTop: '15px',
          padding: '15px',
          backgroundColor: 'rgba(255,255,255,0.1)',
          borderRadius: '8px',
          backdropFilter: 'blur(10px)'
        }}>
          <div style={{ marginBottom: '12px' }}>
            <label style={{
              display: 'block',
              marginBottom: '6px',
              color: 'rgba(255,255,255,0.9)',
              fontWeight: '500',
              fontSize: '14px'
            }}>
              📚 Difficulty Level:
            </label>
            <select
              value={difficultyLevel}
              onChange={handleDifficultyChange}
              style={{
                width: '100%',
                padding: '10px 12px',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '8px',
                backgroundColor: 'rgba(255,255,255,0.95)',
                fontSize: '14px',
                color: '#333',
                transition: 'all 0.3s ease'
              }}
            >
              <option value="default">Based on your profile</option>
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          <div style={{ marginBottom: '12px' }}>
            <label style={{
              display: 'block',
              marginBottom: '6px',
              color: 'rgba(255,255,255,0.9)',
              fontWeight: '500',
              fontSize: '14px'
            }}>
              📝 Content Format:
            </label>
            <select
              value={contentFormat}
              onChange={handleFormatChange}
              style={{
                width: '100%',
                padding: '10px 12px',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '8px',
                backgroundColor: 'rgba(255,255,255,0.95)',
                fontSize: '14px',
                color: '#333',
                transition: 'all 0.3s ease'
              }}
            >
              <option value="default">Default</option>
              <option value="visual">More Visual</option>
              <option value="detailed">More Detailed</option>
              <option value="concise">More Concise</option>
            </select>
          </div>
        </div>
      )}

      <div style={{
        marginTop: '15px',
        padding: '12px',
        backgroundColor: 'rgba(255,255,255,0.1)',
        borderRadius: '8px',
        backdropFilter: 'blur(10px)'
      }}>
        <p style={{
          margin: '0 0 8px 0',
          color: 'rgba(255,255,255,0.9)',
          fontSize: '14px'
        }}>
          <strong style={{ color: 'white' }}>Status:</strong> {isPersonalized ? '🎯 Personalized for your learning profile' : '📋 Using default content'}
        </p>
        {user && (
          <p style={{
            margin: '0',
            color: 'rgba(255,255,255,0.9)',
            fontSize: '14px'
          }}>
            <strong style={{ color: 'white' }}>Your Profile:</strong> {user.softwareExperience} software experience, {user.hardwareExperience} hardware experience
          </p>
        )}
      </div>
    </div>
  );
};

export default ChapterPersonalization;