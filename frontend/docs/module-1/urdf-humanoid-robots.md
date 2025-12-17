---
sidebar_position: 4
---

# URDF for Humanoid Robots

## Chapter Objectives

- Understand the Unified Robot Description Format (URDF)
- Create URDF models for humanoid robots
- Define kinematic chains and joint constraints
- Integrate URDF with ROS 2 simulation

## Introduction to URDF

URDF (Unified Robot Description Format) is an XML format for representing robot models in ROS. It defines the physical and visual properties of a robot including:

- Links (rigid bodies)
- Joints (connections between links)
- Inertial properties
- Visual and collision geometry
- Transmission information

### Why URDF for Humanoid Robots?

Humanoid robots have complex kinematic structures with multiple limbs and joints. URDF provides:

- A standardized way to represent complex robot kinematics
- Integration with ROS tools for visualization and simulation
- Compatibility with kinematics libraries like MoveIt
- Support for dynamics simulation in Gazebo

## Basic URDF Structure

A URDF file has a basic structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links definition -->
  <link name="base_link">
    <!-- Visual and collision properties -->
  </link>

  <!-- Joints definition -->
  <joint name="joint_name" type="revolute">
    <!-- Joint properties -->
  </joint>
</robot>
```

## Link Definition

A link represents a rigid body in the robot:

```xml
<link name="base_link">
  <inertial>
    <mass value="1.0" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1" />
  </inertial>

  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.2 0.2 0.2" />
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1" />
    </material>
  </visual>

  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.2 0.2 0.2" />
    </geometry>
  </collision>
</link>
```

## Joint Definition

Joints connect links and define their relative motion:

```xml
<joint name="hip_joint" type="revolute">
  <parent link="torso" />
  <child link="thigh" />
  <origin xyz="0 0 -0.1" rpy="0 0 0" />
  <axis xyz="0 0 1" />
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1" />
  <dynamics damping="0.1" friction="0.0" />
</joint>
```

## Complete Humanoid Robot URDF

Here's a simplified URDF for a humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">

  <!-- Base Link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 0.5" />
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0" />
    </inertial>
    <visual>
      <origin xyz="0 0 0.5" />
      <geometry>
        <box size="0.3 0.3 1.0" />
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" />
      <geometry>
        <box size="0.3 0.3 1.0" />
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1" />
    </inertial>
    <visual>
      <origin xyz="0 0 0" />
      <geometry>
        <sphere radius="0.15" />
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" />
      <geometry>
        <sphere radius="0.15" />
      </geometry>
    </collision>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="base_link" />
    <child link="head" />
    <origin xyz="0 0 1.0" />
    <axis xyz="0 0 1" />
    <limit lower="-0.785" upper="0.785" effort="10" velocity="1" />
  </joint>

  <!-- Left Arm -->
  <link name="left_shoulder">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 -0.1" />
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05" />
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1" />
      <geometry>
        <cylinder radius="0.05" length="0.2" />
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" />
      <geometry>
        <cylinder radius="0.05" length="0.2" />
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link" />
    <child link="left_shoulder" />
    <origin xyz="0.15 0.1 0.8" />
    <axis xyz="0 1 0" />
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1" />
  </joint>

  <!-- Additional links and joints would continue here -->
  <!-- Right arm, legs, etc. -->

  <!-- Materials -->
  <material name="black">
    <color rgba="0 0 0 1" />
  </material>

  <material name="blue">
    <color rgba="0 0 0.8 1" />
  </material>

  <material name="green">
    <color rgba="0 0.8 0 1" />
  </material>

  <material name="grey">
    <color rgba="0.5 0.5 0.5 1" />
  </material>

  <material name="orange">
    <color rgba="1 0.42 0 1" />
  </material>

  <material name="brown">
    <color rgba="0.87 0.84 0.7 1" />
  </material>

  <material name="red">
    <color rgba="0.8 0 0 1" />
  </material>

  <material name="white">
    <color rgba="1 1 1 1" />
  </material>
</robot>
```

## URDF Tools and Validation

### Checking URDF

```bash
# Validate URDF syntax
check_urdf /path/to/robot.urdf

# Display URDF information
urdf_to_graphiz /path/to/robot.urdf
```

### Visualizing URDF

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat /path/to/robot.urdf)'

# Visualize in RViz
ros2 run rviz2 rviz2
```

## Xacro: XML Macros for URDF

Xacro is a macro language for XML that makes URDF files more maintainable:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="base_width" value="0.3" />
  <xacro:property name="base_height" value="1.0" />

  <!-- Macro for creating a joint -->
  <xacro:macro name="simple_joint" params="name type parent child origin_xyz axis_xyz lower upper">
    <joint name="${name}" type="${type}">
      <parent link="${parent}" />
      <child link="${child}" />
      <origin xyz="${origin_xyz}" />
      <axis xyz="${axis_xyz}" />
      <limit lower="${lower}" upper="${upper}" effort="100" velocity="1" />
    </joint>
  </xacro:macro>

  <!-- Base link using properties -->
  <link name="base_link">
    <inertial>
      <mass value="10.0" />
      <origin xyz="0 0 ${base_height/2}" />
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0" />
    </inertial>
    <visual>
      <origin xyz="0 0 ${base_height/2}" />
      <geometry>
        <box size="${base_width} ${base_width} ${base_height}" />
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1" />
      </material>
    </visual>
  </link>

  <!-- Use the macro -->
  <xacro:simple_joint
    name="test_joint"
    type="revolute"
    parent="base_link"
    child="test_link"
    origin_xyz="0 0 ${base_height}"
    axis_xyz="0 0 1"
    lower="-1.57"
    upper="1.57" />

  <link name="test_link">
    <inertial>
      <mass value="1.0" />
      <origin xyz="0 0 0" />
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05" />
    </inertial>
  </link>

</robot>
```

## Kinematic Chains for Humanoid Robots

Humanoid robots have several kinematic chains:

- **Left Arm Chain**: base_link → left_shoulder → left_elbow → left_wrist
- **Right Arm Chain**: base_link → right_shoulder → right_elbow → right_wrist
- **Left Leg Chain**: base_link → left_hip → left_knee → left_ankle
- **Right Leg Chain**: base_link → right_hip → right_knee → right_ankle
- **Head Chain**: base_link → neck → head

## ROS 2 Integration

### Robot State Publisher

The robot_state_publisher node publishes joint states to TF:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize joint positions
        self.joint_positions = {}

    def joint_state_callback(self, msg):
        """Update joint positions and broadcast transforms"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

        self.broadcast_transforms()

    def broadcast_transforms(self):
        """Broadcast all robot transforms"""
        # Example: broadcast transform for a simple joint
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'head'

        # Calculate transform based on joint position
        head_position = self.joint_positions.get('neck_joint', 0.0)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 1.0 + math.sin(head_position) * 0.1
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
```

## Best Practices for Humanoid URDF

### Link Design

- Use realistic inertial properties
- Include both visual and collision geometry
- Name links consistently (e.g., left_arm_link, right_arm_link)
- Consider the robot's center of mass

### Joint Design

- Set appropriate joint limits based on physical constraints
- Use correct joint types (revolute, prismatic, continuous)
- Include safety margins in joint limits
- Set realistic effort and velocity limits

### Organization

- Separate URDF into multiple files for complex robots
- Use Xacro to reduce duplication
- Include proper materials and colors
- Validate URDF before simulation

## Hands-On Exercise

1. Create a URDF file for a simple humanoid robot with at least 12 joints
2. Use Xacro to make the URDF more maintainable
3. Visualize the robot in RViz
4. Create a simple controller that moves the robot's joints

## Summary

URDF is essential for representing humanoid robots in ROS 2. With proper link and joint definitions, you can create accurate robot models for simulation, visualization, and control. In the next module, we'll explore digital twin technologies for humanoid robots.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic URDF structure and visualization
- **Intermediate**: Dive deeper into kinematic chains and Xacro macros
- **Advanced**: Explore dynamics properties, complex transmissions, and MoveIt integration