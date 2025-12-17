import React, { useState } from 'react';
import RAGChatbot from '@site/src/components/RAGChatbot';

const FloatingChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Floating Chatbot Button */}
      <button
        onClick={toggleChatbot}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          backgroundColor: '#007cba',
          color: 'white',
          border: 'none',
          cursor: 'pointer',
          fontSize: '24px',
          zIndex: '1000',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.3s ease',
        }}
        title="Ask questions about the book content"
      >
        💬
      </button>

      {/* Chatbot Modal */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '20px',
            width: '400px',
            height: '500px',
            zIndex: '1001',
            boxShadow: '0 10px 30px rgba(0,0,0,0.3)',
            borderRadius: '12px',
            overflow: 'hidden',
          }}
        >
          <RAGChatbot
            chapterTitle="Physical AI & Humanoid Robotics Book"
            chapterContent="This is a comprehensive guide on Physical AI & Humanoid Robotics. The book covers four modules: Module 1: The Robotic Nervous System (ROS 2), Module 2: The Digital Twin (Gazebo & Unity), Module 3: The AI-Robot Brain (NVIDIA Isaac), and Module 4: Vision-Language-Action (VLA). You can ask questions about any of these topics."
          />
          <button
            onClick={toggleChatbot}
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              background: 'rgba(0,0,0,0.5)',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '30px',
              height: '30px',
              cursor: 'pointer',
              fontSize: '16px',
              zIndex: '1002',
            }}
          >
            ×
          </button>
        </div>
      )}

      {/* Overlay when chatbot is open */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: 'rgba(0,0,0,0.5)',
            zIndex: '999',
          }}
          onClick={toggleChatbot}
        />
      )}
    </>
  );
};

export default FloatingChatbot;