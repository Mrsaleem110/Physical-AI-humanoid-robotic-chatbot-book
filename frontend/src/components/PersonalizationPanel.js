import React, { useState, useEffect } from 'react';

const PersonalizationPanel = () => {
  const [preferences, setPreferences] = useState({
    difficulty_level: 'intermediate',
    learning_style: 'visual',
    content_format: 'mixed',
    update_frequency: 'daily',
    notification_preferences: {
      email: true,
      push: true,
      sms: false
    },
    adaptive_preferences: {
      show_hints: true,
      provide_examples: true,
      extra_practice: false
    },
    enable_personalization: true,
    enable_adaptive_content: true
  });

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const handleInputChange = (field, value) => {
    setPreferences(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleNestedChange = (parentField, childField, value) => {
    setPreferences(prev => ({
      ...prev,
      [parentField]: {
        ...(prev[parentField]),
        [childField]: value
      }
    }));
  };

  const handleSavePreferences = async () => {
    try {
      setSaving(true);
      // In a real implementation, this would save to your backend API
      // For now, we'll just simulate the save
      await new Promise(resolve => setTimeout(resolve, 500));
      alert('Preferences saved successfully!');
    } catch (error) {
      console.error('Error saving preferences:', error);
      alert('Error saving preferences. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  // CSS styles for the personalization panel
  const styles = {
    panel: {
      border: '1px solid #ddd',
      borderRadius: '8px',
      padding: '16px',
      margin: '1rem 0',
      backgroundColor: 'white',
      fontFamily: 'system-ui, sans-serif',
      maxWidth: '600px',
    },
    section: {
      marginBottom: '16px',
      padding: '12px',
      border: '1px solid #eee',
      borderRadius: '4px',
    },
    sectionTitle: {
      fontSize: '1.1rem',
      fontWeight: 'bold',
      marginBottom: '12px',
      color: '#282c34',
    },
    group: {
      marginBottom: '8px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    },
    label: {
      fontSize: '0.9rem',
      marginRight: '8px',
    },
    select: {
      padding: '4px 8px',
      border: '1px solid #ccc',
      borderRadius: '4px',
      fontSize: '0.9rem',
    },
    checkboxLabel: {
      display: 'flex',
      alignItems: 'center',
      fontSize: '0.9rem',
    },
    checkbox: {
      marginRight: '6px',
    },
    saveButton: {
      marginTop: '16px',
      padding: '8px 16px',
      backgroundColor: '#007cba',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.9rem',
    },
    title: {
      fontSize: '1.2rem',
      fontWeight: 'bold',
      marginBottom: '12px',
      color: '#282c34',
    },
  };

  return (
    <div style={styles.panel}>
      <h3 style={styles.title}>Personalization Settings</h3>

      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Learning Preferences</h4>

        <div style={styles.group}>
          <label style={styles.label}>Difficulty Level:</label>
          <select
            value={preferences.difficulty_level}
            onChange={(e) => handleInputChange('difficulty_level', e.target.value)}
            style={styles.select}
          >
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>

        <div style={styles.group}>
          <label style={styles.label}>Learning Style:</label>
          <select
            value={preferences.learning_style}
            onChange={(e) => handleInputChange('learning_style', e.target.value)}
            style={styles.select}
          >
            <option value="visual">Visual</option>
            <option value="auditory">Auditory</option>
            <option value="kinesthetic">Kinesthetic</option>
          </select>
        </div>

        <div style={styles.group}>
          <label style={styles.label}>Content Format:</label>
          <select
            value={preferences.content_format}
            onChange={(e) => handleInputChange('content_format', e.target.value)}
            style={styles.select}
          >
            <option value="text">Text</option>
            <option value="video">Video</option>
            <option value="interactive">Interactive</option>
            <option value="mixed">Mixed</option>
          </select>
        </div>
      </div>

      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Notification Preferences</h4>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.notification_preferences.email}
              onChange={(e) => handleNestedChange('notification_preferences', 'email', e.target.checked)}
              style={styles.checkbox}
            />
            Email Notifications
          </label>
        </div>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.notification_preferences.push}
              onChange={(e) => handleNestedChange('notification_preferences', 'push', e.target.checked)}
              style={styles.checkbox}
            />
            Push Notifications
          </label>
        </div>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.notification_preferences.sms}
              onChange={(e) => handleNestedChange('notification_preferences', 'sms', e.target.checked)}
              style={styles.checkbox}
            />
            SMS Notifications
          </label>
        </div>
      </div>

      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>Adaptive Learning</h4>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.adaptive_preferences.show_hints}
              onChange={(e) => handleNestedChange('adaptive_preferences', 'show_hints', e.target.checked)}
              style={styles.checkbox}
            />
            Show Hints
          </label>
        </div>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.adaptive_preferences.provide_examples}
              onChange={(e) => handleNestedChange('adaptive_preferences', 'provide_examples', e.target.checked)}
              style={styles.checkbox}
            />
            Provide Examples
          </label>
        </div>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.adaptive_preferences.extra_practice}
              onChange={(e) => handleNestedChange('adaptive_preferences', 'extra_practice', e.target.checked)}
              style={styles.checkbox}
            />
            Extra Practice Problems
          </label>
        </div>
      </div>

      <div style={styles.section}>
        <h4 style={styles.sectionTitle}>General Settings</h4>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.enable_personalization}
              onChange={(e) => handleInputChange('enable_personalization', e.target.checked)}
              style={styles.checkbox}
            />
            Enable Personalization
          </label>
        </div>

        <div style={styles.group}>
          <label style={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={preferences.enable_adaptive_content}
              onChange={(e) => handleInputChange('enable_adaptive_content', e.target.checked)}
              style={styles.checkbox}
            />
            Enable Adaptive Content
          </label>
        </div>
      </div>

      <button
        onClick={handleSavePreferences}
        disabled={saving}
        style={styles.saveButton}
      >
        {saving ? 'Saving...' : 'Save Preferences'}
      </button>
    </div>
  );
};

export default PersonalizationPanel;