import React, { useState, useEffect, useRef } from 'react';
import { useUser } from '../contexts/UserContext';

const RAGChatbot = ({ chapterContent, chapterTitle }) => {
  const { user } = useUser();
  const [messages, setMessages] = useState([
    {
      id: '1',
      text: `Hello! I'm your AI assistant for "${chapterTitle}". I can answer questions about this chapter's content. How can I help you?`,
      sender: 'bot',
      timestamp: new Date(),
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Knowledge base for modules
  const moduleKnowledgeBase = {
    "module-1": {
      title: "Introduction to ROS 2 and Humanoid Robotics",
      content: `Module 1 covers the fundamentals of Robot Operating System 2 (ROS 2) and its application to humanoid robotics.
      Key topics include: ROS 2 architecture, nodes, topics, services, actions, URDF (Unified Robot Description Format) for humanoid robots,
      Python-based ROS control systems, and basic robot modeling. This module provides the foundational knowledge needed to work with
      humanoid robots using ROS 2, including how to create publishers, subscribers, services, and how to model humanoid robots with URDF files.`,
      chapters: [
        { name: "Introduction to ROS 2", description: "Basic concepts of ROS 2 architecture" },
        { name: "ROS 2 Nodes, Topics, Services", description: "Communication patterns in ROS 2" },
        { name: "Python ROS Control", description: "Controlling robots with Python" },
        { name: "URDF for Humanoid Robots", description: "Modeling humanoid robots" }
      ]
    },
    "module-2": {
      title: "Simulation and Physics Modeling",
      content: `Module 2 focuses on simulation environments and physics modeling for humanoid robots.
      Topics include: Physics simulation fundamentals, gravity and collision modeling, sensor simulation,
      Unity-based HRI (Human-Robot Interaction) visualization, and realistic environment modeling.
      Students learn to create realistic simulation environments for humanoid robots, implement physics
      constraints, model sensors, and visualize robot interactions in Unity. The module also covers
      collision detection and response systems critical for humanoid robot simulation.`,
      chapters: [
        { name: "Physics Simulation Fundamentals", description: "Basic physics in robotics simulation" },
        { name: "Gravity and Collision Modeling", description: "Modeling physical interactions" },
        { name: "Sensor Simulation", description: "Simulating robot sensors" },
        { name: "Unity HRI Visualization", description: "Human-robot interaction visualization" }
      ]
    },
    "module-3": {
      title: "Navigation and Path Planning",
      content: `Module 3 covers navigation systems and path planning algorithms for humanoid robots.
      Key topics include: Robot navigation stacks, path planning algorithms (A*, Dijkstra, RRT),
      obstacle avoidance, SLAM (Simultaneous Localization and Mapping), and humanoid-specific
      navigation challenges. Students learn to implement navigation systems that account for
      the unique kinematics and dynamics of humanoid robots, including bipedal locomotion planning
      and terrain adaptation strategies.`,
      chapters: [
        { name: "Navigation Fundamentals", description: "Basic navigation concepts" },
        { name: "Path Planning Algorithms", description: "A*, Dijkstra, RRT algorithms" },
        { name: "SLAM Systems", description: "Localization and mapping" },
        { name: "Humanoid Locomotion Planning", description: "Bipedal movement planning" }
      ]
    },
    "module-4": {
      title: "AI and Machine Learning for Humanoid Robots",
      content: `Module 4 explores AI and machine learning applications in humanoid robotics.
      Topics include: Vision-Language-Action (VLA) models, reinforcement learning for humanoid control,
      computer vision for humanoid robots, natural language processing for HRI, and deep learning
      applications. Students learn to implement AI systems that enable humanoid robots to perceive
      their environment, understand human commands, and execute complex tasks using machine learning
      techniques specifically tailored for humanoid robot platforms.`,
      chapters: [
        { name: "Vision-Language-Action Models", description: "VLA for humanoid robots" },
        { name: "Reinforcement Learning", description: "Learning-based control" },
        { name: "Computer Vision", description: "Visual perception systems" },
        { name: "Natural Language Processing", description: "Human-robot communication" }
      ]
    }
  };

  // Function to detect if question is about modules or physical AI & humanoid robotics
  const detectModuleQuestion = (question) => {
    const questionLower = question.toLowerCase();

    // Check for physical AI & humanoid robotics queries
    if (questionLower.includes('physical ai') || questionLower.includes('humanoid robotics') || questionLower.includes('physical ai & humanoid robotics')) {
      return 'physical-ai-humanoid';
    }

    // Check for module references
    if (questionLower.includes('module 1') || questionLower.includes('module1') || questionLower.includes('first module')) {
      return 'module-1';
    } else if (questionLower.includes('module 2') || questionLower.includes('module2') || questionLower.includes('second module')) {
      return 'module-2';
    } else if (questionLower.includes('module 3') || questionLower.includes('module3') || questionLower.includes('third module')) {
      return 'module-3';
    } else if (questionLower.includes('module 4') || questionLower.includes('module4') || questionLower.includes('fourth module')) {
      return 'module-4';
    } else if (questionLower.includes('module') && (questionLower.includes('what') || questionLower.includes('tell me about') || questionLower.includes('describe') || questionLower.includes('explain') || (questionLower.includes('are') && questionLower.includes('the')))) {
      // Check if question is asking about modules in general
      return 'general-module';
    }

    return null;
  };

  // Function to get relevant context from chapter content based on user question
  const getRelevantContext = (question) => {
    // First check if this is a module-specific question
    const moduleMatch = detectModuleQuestion(question);
    if (moduleMatch && moduleKnowledgeBase[moduleMatch]) {
      return moduleKnowledgeBase[moduleMatch].content;
    }

    // Simple keyword matching to find relevant sections in the chapter
    const keywords = question.toLowerCase().split(' ').filter(word => word.length > 3);
    let relevantContent = chapterContent || '';

    // Look for sentences that contain the keywords
    keywords.forEach(keyword => {
      if (relevantContent.toLowerCase().includes(keyword)) {
        // Extract the sentence containing the keyword
        const sentences = relevantContent.split(/(?<=[.!?])\s+/);
        const matchingSentence = sentences.find(sentence =>
          sentence.toLowerCase().includes(keyword)
        );
        if (matchingSentence) {
          relevantContent = matchingSentence;
        }
      }
    });

    return relevantContent.substring(0, 500); // Limit context length
  };

  const generateBotResponse = (userQuestion) => {
    // In a real RAG implementation, this would query a vector database
    // For this mock, we'll use simple keyword matching and predefined responses

    const questionLower = userQuestion.toLowerCase();
    const moduleMatch = detectModuleQuestion(userQuestion);

    // Handle physical AI & humanoid robotics queries first
    if (moduleMatch === 'physical-ai-humanoid') {
      return `Physical AI & Humanoid Robotics is an interdisciplinary field that combines artificial intelligence with physical robotic systems, specifically focusing on humanoid robots.

The curriculum covers:
- **Module 1**: Introduction to ROS 2 and Humanoid Robotics - foundational concepts, ROS 2 architecture, and robot modeling
- **Module 2**: Simulation and Physics Modeling - physics simulation, sensor modeling, and visualization
- **Module 3**: Navigation and Path Planning - locomotion, navigation algorithms, and movement planning
- **Module 4**: AI and Machine Learning - vision-language-action models, learning algorithms, and intelligent control

This field aims to create robots that can interact with the physical world in human-like ways, using AI to perceive, reason, and act in complex environments.`;
    }

    // Handle module-specific questions next
    if (moduleMatch) {
      // If asking about modules in general
      if (moduleMatch === 'general-module') {
        return `We have 4 modules in the humanoid robotics curriculum:

1. Module 1: ${moduleKnowledgeBase['module-1'].title}
   - ${moduleKnowledgeBase['module-1'].chapters.length} chapters covering ROS 2 fundamentals

2. Module 2: ${moduleKnowledgeBase['module-2'].title}
   - ${moduleKnowledgeBase['module-2'].chapters.length} chapters covering simulation and physics

3. Module 3: ${moduleKnowledgeBase['module-3'].title}
   - ${moduleKnowledgeBase['module-3'].chapters.length} chapters covering navigation and path planning

4. Module 4: ${moduleKnowledgeBase['module-4'].title}
   - ${moduleKnowledgeBase['module-4'].chapters.length} chapters covering AI and machine learning

You can ask specific questions about any module (e.g., "What is in module 1?" or "Tell me about module 2").`;
      }
      // If asking for specific module info
      else if (moduleKnowledgeBase[moduleMatch]) {
        const moduleInfo = moduleKnowledgeBase[moduleMatch];
        return `Here's information about ${moduleInfo.title}:

${moduleInfo.content}

Key chapters in this module:
${moduleInfo.chapters.map((ch, idx) => `${idx + 1}. ${ch.name} - ${ch.description}`).join('\n')}`;
      }
    }

    // Check for other predefined responses
    const relevantContext = getRelevantContext(userQuestion);

    // Predefined responses for common robotics topics
    if (questionLower.includes('ros') || questionLower.includes('robot operating system')) {
      return "The Robot Operating System (ROS) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.";
    } else if (questionLower.includes('gazebo') || questionLower.includes('simulation')) {
      return "Gazebo is a robot simulation environment that provides the ability to accurately and efficiently simulate populations of robots in complex indoor and outdoor environments. It provides the tools and infrastructure needed for robot simulation, including physics simulation, sensor simulation, and robot programming interface.";
    } else if (questionLower.includes('isaac') || questionLower.includes('nvidia')) {
      return "NVIDIA Isaac is a robotics platform that includes a simulation application and AI training framework based on the Omniverse platform. It provides developers with the tools to design, simulate, train, and deploy AI-based solutions for robotics applications.";
    } else if (questionLower.includes('vision') || questionLower.includes('language') || questionLower.includes('action')) {
      return "Vision-Language-Action (VLA) models combine visual perception, natural language understanding, and action generation. These models enable robots to understand complex human instructions and perform appropriate physical actions in response.";
    } else if (questionLower.includes('urdf') || questionLower.includes('model')) {
      return "URDF (Unified Robot Description Format) is an XML format used to describe robot models in ROS. It defines the physical and visual properties of a robot, including links, joints, and their relationships.";
    } else if (questionLower.includes('navigation') || questionLower.includes('path')) {
      return "Robot navigation involves the process of safely moving a robot from one location to another. This typically involves perception of the environment, path planning, and path execution using the ROS Navigation Stack.";
    } else {
      // If no specific topic is detected, generate a response based on the relevant context
      if (relevantContext && relevantContext.length > 0) {
        return `Based on the chapter content, here's information related to your question: "${relevantContext}". For more details, please refer to the specific section in the chapter.`;
      } else {
        return `I can help answer questions about "${chapterTitle}". Please ask a specific question about the content, and I'll do my best to provide an answer based on the chapter material.`;
      }
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Generate bot response based on chapter content
      const botResponse = generateBotResponse(inputValue);

      // Add bot response
      const botMessage = {
        id: Date.now().toString(),
        text: botResponse,
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message
      const errorMessage = {
        id: Date.now().toString(),
        text: 'Sorry, I encountered an error processing your question. Please try again.',
        sender: 'bot',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleUseSelectedText = () => {
    if (selectedText) {
      setInputValue(selectedText);
      // Clear the selection
      if (window.getSelection) {
        window.getSelection().removeAllRanges();
      }
    }
  };

  // Function to capture selected text
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Simple CSS styles for the chatbot
  const styles = {
    chatWidget: {
      border: '1px solid #ddd',
      borderRadius: '8px',
      overflow: 'hidden',
      maxWidth: '100%',
      height: '500px',
      display: 'flex',
      flexDirection: 'column',
      margin: '1rem 0',
      fontFamily: 'system-ui, sans-serif',
    },
    chatHeader: {
      backgroundColor: '#282c34',
      color: 'white',
      padding: '12px',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    chatControls: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    chatMessages: {
      flex: 1,
      padding: '12px',
      overflowY: 'auto',
      backgroundColor: '#f5f5f5',
    },
    message: {
      marginBottom: '12px',
      maxWidth: '80%',
    },
    userMessage: {
      marginLeft: 'auto',
      backgroundColor: '#007cba',
      color: 'white',
      padding: '8px',
      borderRadius: '8px',
    },
    botMessage: {
      marginRight: 'auto',
      backgroundColor: '#e9ecef',
      padding: '8px',
      borderRadius: '8px',
    },
    messageContent: {
      margin: 0,
    },
    timestamp: {
      fontSize: '0.7em',
      color: '#888',
      marginTop: '4px',
      display: 'block',
    },
    typingIndicator: {
      display: 'flex',
      alignItems: 'center',
    },
    typingDot: {
      width: '8px',
      height: '8px',
      backgroundColor: '#888',
      borderRadius: '50%',
      margin: '0 2px',
      animation: 'typing 1.4s infinite ease-in-out',
    },
    chatInputArea: {
      display: 'flex',
      padding: '12px',
      backgroundColor: 'white',
      borderTop: '1px solid #ddd',
    },
    chatInput: {
      flex: 1,
      padding: '8px',
      border: '1px solid #ddd',
      borderRadius: '4px',
      resize: 'vertical',
    },
    sendButton: {
      marginLeft: '8px',
      padding: '8px 12px',
      backgroundColor: '#007cba',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
    },
    selectedTextNotice: {
      padding: '8px',
      backgroundColor: '#fff3cd',
      border: '1px solid #ffeaa7',
      borderRadius: '4px',
      marginBottom: '8px',
      fontSize: '0.9em',
    },
    useTextButton: {
      padding: '4px 8px',
      backgroundColor: '#007cba',
      color: 'white',
      border: 'none',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '0.8rem',
      marginLeft: '8px',
    },
  };

  return (
    <div style={{
      border: '1px solid #ddd',
      borderRadius: '12px',
      overflow: 'hidden',
      maxWidth: '100%',
      height: '500px',
      display: 'flex',
      flexDirection: 'column',
      margin: '1rem 0',
      fontFamily: 'system-ui, sans-serif',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <div style={{
        backgroundColor: 'rgba(255,255,255,0.1)',
        color: 'white',
        padding: '15px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backdropFilter: 'blur(10px)'
      }}>
        <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
          🤖 Chapter Assistant
        </h3>
        <div style={{ fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            {user ? `👋 ${user.name || 'User'}` : '🔐 Log in for personalized experience'}
          </span>
        </div>
      </div>

      {selectedText && (
        <div style={{
          padding: '10px',
          backgroundColor: 'rgba(255, 193, 7, 0.2)',
          border: '1px solid rgba(255, 193, 7, 0.3)',
          borderRadius: '4px',
          marginBottom: '8px',
          fontSize: '0.9em',
          backdropFilter: 'blur(10px)'
        }}>
          <span style={{ color: 'rgba(255,255,255,0.9)' }}>
            📝 Selected text: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"
          </span>
          <button
            onClick={handleUseSelectedText}
            style={{
              padding: '4px 8px',
              backgroundColor: 'rgba(255,255,255,0.2)',
              color: 'white',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.8rem',
              marginLeft: '8px',
              transition: 'all 0.3s ease'
            }}
          >
            💬 Use in Question
          </button>
        </div>
      )}

      <div style={{
        flex: 1,
        padding: '15px',
        overflowY: 'auto',
        backgroundColor: 'rgba(255,255,255,0.05)',
        backdropFilter: 'blur(10px)'
      }}>
        {messages.map((message) => (
          <div
            key={message.id}
            style={{
              marginBottom: '12px',
              maxWidth: '80%',
              marginLeft: message.sender === 'user' ? 'auto' : '0',
              marginRight: message.sender === 'user' ? '0' : 'auto',
              backgroundColor: message.sender === 'user' ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.85)',
              padding: '12px',
              borderRadius: '12px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
            }}
          >
            <div style={{ margin: 0 }}>
              <p style={{ margin: '0 0 8px 0', color: '#333' }}>{message.text}</p>
              <span style={{ fontSize: '0.7em', color: '#666', display: 'block', textAlign: 'right' }}>
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </div>
        ))}
        {isLoading && (
          <div style={{
            marginBottom: '12px',
            maxWidth: '80%',
            backgroundColor: 'rgba(255,255,255,0.85)',
            padding: '12px',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <div style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <span style={{ width: '8px', height: '8px', backgroundColor: '#666', borderRadius: '50%', margin: '0 2px', animation: 'typing 1.4s infinite' }}></span>
                <span style={{ width: '8px', height: '8px', backgroundColor: '#666', borderRadius: '50%', margin: '0 2px', animation: 'typing 1.4s infinite', animationDelay: '0.2s' }}></span>
                <span style={{ width: '8px', height: '8px', backgroundColor: '#666', borderRadius: '50%', margin: '0 2px', animation: 'typing 1.4s infinite', animationDelay: '0.4s' }}></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div style={{
        display: 'flex',
        padding: '15px',
        backgroundColor: 'rgba(255,255,255,0.1)',
        borderTop: '1px solid rgba(255,255,255,0.2)',
        backdropFilter: 'blur(10px)'
      }}>
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="💬 Ask a question about this chapter..."
          disabled={isLoading}
          rows={2}
          style={{
            flex: 1,
            padding: '12px',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            resize: 'vertical',
            fontSize: '0.9rem',
            backgroundColor: 'rgba(255,255,255,0.95)',
            color: '#333',
            transition: 'all 0.3s ease'
          }}
        />
        <button
          onClick={handleSendMessage}
          disabled={!inputValue.trim() || isLoading}
          style={{
            marginLeft: '10px',
            padding: '12px 16px',
            backgroundColor: isLoading ? '#6c757d' : '#007cba',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '0.9rem',
            fontWeight: '600',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}
        >
          {isLoading ? '...' : 'Send'}
        </button>
      </div>

      <style jsx>{`
        @keyframes typing {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }
      `}</style>
    </div>
  );
};

export default RAGChatbot;