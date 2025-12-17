// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Introduction',
      items: ['intro', 'getting-started', 'interactive-features'],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/intro-to-ros2',
        'module-1/ros2-nodes-topics-services',
        'module-1/python-ros-control',
        'module-1/urdf-humanoid-robots'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/physics-simulation-fundamentals',
        'module-2/gravity-collision-modeling',
        'module-2/unity-hri-visualization',
        'module-2/sensor-simulation'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3/isaac-sim-photorealistic',
        'module-3/synthetic-data-generation',
        'module-3/isaac-ros-vslam-perception',
        'module-3/nav2-humanoid-locomotion'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4/whisper-voice-intent',
        'module-4/llm-cognitive-planning',
        'module-4/object-detection-manipulation',
        'module-4/capstone-autonomous-humanoid'
      ],
      collapsed: false,
    }
  ],
};

module.exports = sidebars;