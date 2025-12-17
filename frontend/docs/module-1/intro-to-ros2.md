---
sidebar_position: 1
---

# Introduction to ROS 2

## Chapter Objectives

- Understand the fundamentals of ROS 2 architecture
- Learn about the robotic nervous system concept
- Set up your ROS 2 development environment
- Create your first ROS 2 workspace

## What is ROS 2?

ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Concepts

- **Middleware**: ROS 2 uses DDS (Data Distribution Service) as its communication middleware
- **Packages**: Organize code and resources into reusable units
- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/reply communication
- **Actions**: Goal-oriented communication with feedback

### Why ROS 2 for Humanoid Robots?

Humanoid robots require complex coordination between multiple subsystems:

- Motor control systems
- Sensor processing
- Perception modules
- Planning and control
- Human-robot interaction

ROS 2 provides the infrastructure to connect these subsystems efficiently while maintaining modularity.

## Setting Up ROS 2

### Installation

```bash
# Update package lists
sudo apt update

# Install ROS 2 Iron (latest LTS)
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install ros-iron-desktop
sudo apt install ros-dev-tools
```

### Environment Setup

```bash
# Source ROS 2 environment
source /opt/ros/iron/setup.bash

# Add to your bashrc to source automatically
echo "source /opt/ros/iron/setup.bash" >> ~/.bashrc
```

## Creating Your First Workspace

### Workspace Structure

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

### Creating a Package

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_package
```

## The Robotic Nervous System

Think of ROS 2 as the nervous system of your robot:

- **Sensory Input**: Sensors publish data to topics (like sensory neurons)
- **Processing Centers**: Nodes process information and make decisions (like the brain)
- **Motor Output**: Actuators receive commands from services/actions (like motor neurons)
- **Communication Pathways**: Topics, services, and actions serve as neural pathways

## Hands-On Exercise

1. Create a new ROS 2 workspace
2. Create a simple package
3. Verify your installation with `ros2 topic list`

## Summary

ROS 2 provides the foundational communication infrastructure for humanoid robots. Understanding its architecture is crucial for building complex robotic behaviors. In the next chapter, we'll explore nodes, topics, and services in detail.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on installation and basic workspace creation
- **Intermediate**: Dive deeper into package structure and build systems
- **Advanced**: Explore different DDS implementations and real-time considerations