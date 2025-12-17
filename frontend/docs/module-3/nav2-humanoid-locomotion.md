---
sidebar_position: 3
---

# Nav2 for Humanoid Locomotion and Path Planning

## Chapter Objectives

- Understand Nav2 architecture and its adaptation for humanoid robots
- Implement path planning algorithms for bipedal locomotion
- Configure Nav2 for humanoid-specific navigation challenges
- Integrate whole-body motion planning with navigation
- Optimize navigation for real-time humanoid locomotion

## Introduction to Nav2 for Humanoid Robots

Navigation2 (Nav2) is ROS 2's state-of-the-art navigation framework that provides path planning, obstacle avoidance, and localization capabilities. For humanoid robots, Nav2 requires special adaptations to handle:

- **Bipedal Locomotion**: Different motion constraints compared to wheeled robots
- **Dynamic Balance**: Need to maintain balance during navigation
- **Footstep Planning**: Complex gait patterns and foot placement
- **Whole-Body Planning**: Integration of navigation with full-body motion
- **Stability Constraints**: Maintaining center of mass within support polygon

### Key Challenges for Humanoid Navigation

1. **Non-holonomic Constraints**: Limited movement compared to omnidirectional robots
2. **Balance Maintenance**: Continuous balance during motion execution
3. **Footstep Planning**: Discrete foot placement planning
4. **Dynamic Obstacle Avoidance**: Avoiding obstacles while maintaining balance
5. **Multi-terrain Navigation**: Adapting to different surface types

## Nav2 Architecture for Humanoid Robots

### Modified Nav2 Stack Components

```python
# python/humanoid_nav2_components.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class HumanoidNav2Node(Node):
    def __init__(self):
        super().__init__('humanoid_nav2_node')

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers and subscribers
        self.global_plan_pub = self.create_publisher(Path, '/humanoid/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/humanoid/local_plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/humanoid/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/humanoid/nav_status', 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/humanoid/goal',
            self.goal_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Humanoid-specific state
        self.current_pose = None
        self.current_velocity = None
        self.goal_pose = None
        self.navigation_state = "IDLE"  # IDLE, PLANNING, EXECUTING, RECOVERY
        self.path = []
        self.current_waypoint = 0

        # Humanoid locomotion parameters
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters (for stepping over small obstacles)
        self.max_step_up = 0.1  # maximum step up height
        self.max_step_down = 0.1 # maximum step down height
        self.turn_angle = 0.2   # radians per step for turning

        # Navigation parameters
        self.linear_vel = 0.1   # m/s
        self.angular_vel = 0.2  # rad/s
        self.arrival_threshold = 0.2  # meters
        self.yaw_threshold = 0.1      # radians

        # Path planning components
        self.global_planner = HumanoidGlobalPlanner()
        self.local_planner = HumanoidLocalPlanner()
        self.footstep_planner = FootstepPlanner()

        self.get_logger().info("Humanoid Nav2 Node initialized")

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def goal_callback(self, msg):
        """Handle new navigation goal"""
        self.goal_pose = msg.pose

        # Check if we have current pose
        if self.current_pose is not None:
            self.navigation_state = "PLANNING"
            self.plan_path()

    def scan_callback(self, msg):
        """Process laser scan for local planning"""
        if self.navigation_state == "EXECUTING":
            # Update local plan based on obstacles
            self.update_local_plan(msg)

    def plan_path(self):
        """Plan global path to goal"""
        if self.current_pose is None or self.goal_pose is None:
            return

        # Convert poses to numpy arrays for planning
        start = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.get_yaw_from_quaternion(self.current_pose.orientation)
        ])

        goal = np.array([
            self.goal_pose.position.x,
            self.goal_pose.position.y,
            self.get_yaw_from_quaternion(self.goal_pose.orientation)
        ])

        # Plan global path
        global_path = self.global_planner.plan(start, goal)

        if global_path is not None:
            # Convert path to footstep plan
            footstep_plan = self.footstep_planner.plan_footsteps(global_path)

            # Publish global plan
            self.publish_global_plan(footstep_plan)

            # Start execution
            self.path = footstep_plan
            self.current_waypoint = 0
            self.navigation_state = "EXECUTING"

            # Start executing the plan
            self.execute_path()

    def update_local_plan(self, scan_msg):
        """Update local plan based on sensor data"""
        if self.navigation_state != "EXECUTING":
            return

        # Check for obstacles in current path
        obstacles = self.process_scan_for_obstacles(scan_msg)

        if self.local_planner.need_replanning(obstacles, self.path, self.current_waypoint):
            self.get_logger().info("Replanning local path due to obstacles")

            # Create temporary goal at current position + lookahead
            current_pos = self.get_current_position()
            lookahead_goal = self.get_lookahead_goal(current_pos)

            local_path = self.local_planner.plan_local(
                current_pos, lookahead_goal, obstacles
            )

            if local_path is not None:
                self.path = local_path
                self.current_waypoint = 0

    def execute_path(self):
        """Execute the planned path"""
        if not self.path or self.current_waypoint >= len(self.path):
            self.navigation_state = "IDLE"
            self.publish_status("GOAL_REACHED")
            return

        # Get next waypoint
        target_waypoint = self.path[self.current_waypoint]

        # Calculate required motion
        cmd_vel = self.calculate_motion_to_waypoint(target_waypoint)

        # Check if reached waypoint
        if self.is_at_waypoint(target_waypoint):
            self.current_waypoint += 1
            if self.current_waypoint >= len(self.path):
                self.navigation_state = "IDLE"
                self.publish_status("GOAL_REACHED")
                return

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

    def calculate_motion_to_waypoint(self, waypoint):
        """Calculate motion command to reach waypoint"""
        cmd = Twist()

        if self.current_pose is None:
            return cmd

        # Calculate distance and angle to waypoint
        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate desired heading
        desired_yaw = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

        # Calculate angular error
        angle_error = self.normalize_angle(desired_yaw - current_yaw)

        # Set velocities based on errors
        if distance > self.arrival_threshold:
            cmd.linear.x = min(self.linear_vel, distance * 2.0)  # Proportional control
        else:
            cmd.linear.x = 0.0

        if abs(angle_error) > self.yaw_threshold:
            cmd.angular.z = max(-self.angular_vel, min(self.angular_vel, angle_error * 2.0))
        else:
            cmd.angular.z = 0.0

        return cmd

    def is_at_waypoint(self, waypoint):
        """Check if robot is at the specified waypoint"""
        if self.current_pose is None:
            return False

        dx = waypoint[0] - self.current_pose.position.x
        dy = waypoint[1] - self.current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        return distance <= self.arrival_threshold

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_current_position(self):
        """Get current position as numpy array"""
        if self.current_pose is not None:
            return np.array([
                self.current_pose.position.x,
                self.current_pose.position.y
            ])
        return np.array([0.0, 0.0])

    def get_lookahead_goal(self, current_pos):
        """Get goal position for local planning"""
        if self.current_waypoint < len(self.path):
            return np.array(self.path[self.current_waypoint][:2])
        elif self.goal_pose is not None:
            return np.array([
                self.goal_pose.position.x,
                self.goal_pose.position.y
            ])
        return current_pos

    def process_scan_for_obstacles(self, scan_msg):
        """Process laser scan to detect obstacles"""
        obstacles = []

        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        for i, range_val in enumerate(scan_msg.ranges):
            if not (math.isnan(range_val) or math.isinf(range_val)) and range_val < 1.0:  # Within 1m
                angle = angle_min + i * angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                obstacles.append((x, y, range_val))

        return obstacles

    def publish_global_plan(self, path):
        """Publish global path for visualization"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.global_plan_pub.publish(path_msg)

    def publish_status(self, status):
        """Publish navigation status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

class HumanoidGlobalPlanner:
    """Global planner adapted for humanoid robots"""
    def __init__(self):
        # Initialize global planner (e.g., A* or Dijkstra)
        pass

    def plan(self, start, goal):
        """Plan global path from start to goal"""
        # For humanoid robots, we might use a grid-based planner
        # that considers walkable areas and step constraints
        path = self.a_star_plan(start, goal)
        return path

    def a_star_plan(self, start, goal):
        """A* path planning algorithm"""
        # Simplified A* implementation
        # In practice, this would be more complex and consider humanoid constraints
        path = [start, goal]  # Simplified for example
        return path

class HumanoidLocalPlanner:
    """Local planner for humanoid robots"""
    def __init__(self):
        # Initialize local planner (e.g., DWA or TEB)
        pass

    def plan_local(self, current_pos, goal_pos, obstacles):
        """Plan local path considering obstacles"""
        # For humanoid robots, consider step-by-step planning
        # that accounts for balance and foot placement
        local_path = [current_pos, goal_pos]  # Simplified
        return local_path

    def need_replanning(self, obstacles, path, current_waypoint):
        """Check if replanning is needed"""
        # Check if obstacles block current path
        for obs_x, obs_y, obs_dist in obstacles:
            if obs_dist < 0.5:  # Within 50cm
                return True
        return False

class FootstepPlanner:
    """Plan footstep sequences for humanoid navigation"""
    def __init__(self):
        self.step_length = 0.3
        self.step_width = 0.2
        self.max_turn = 0.3  # radians

    def plan_footsteps(self, path):
        """Convert path to footstep plan"""
        footsteps = []

        if len(path) < 2:
            return footsteps

        # Convert path to footstep sequence
        # This is a simplified approach - real implementation would be more complex
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]

            # Calculate intermediate footsteps
            dist = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            steps_needed = max(1, int(dist / self.step_length))

            for j in range(steps_needed):
                ratio = j / steps_needed
                x = start[0] + ratio * (end[0] - start[0])
                y = start[1] + ratio * (end[1] - start[1])
                theta = start[2] if len(start) > 2 else 0.0

                footsteps.append([x, y, theta])

        return footsteps

def main(args=None):
    rclpy.init(args=args)
    nav2_node = HumanoidNav2Node()

    try:
        rclpy.spin(nav2_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav2_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Footstep Planning Algorithms

### Basic Footstep Planning

```python
# python/footstep_planning.py
import numpy as np
import math
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

class FootstepPlanner:
    def __init__(self, step_length=0.3, step_width=0.2, step_height=0.05):
        self.step_length = step_length
        self.step_width = step_width
        self.step_height = step_height

        # Support polygon parameters
        self.foot_separation = step_width
        self.foot_length = 0.25
        self.foot_width = 0.15

    def plan_footsteps(self, start_pose, goal_pose, terrain_map=None):
        """
        Plan footsteps from start to goal considering terrain
        start_pose: [x, y, theta]
        goal_pose: [x, y, theta]
        """
        footsteps = []

        # Calculate straight-line path
        dx = goal_pose[0] - start_pose[0]
        dy = goal_pose[1] - start_pose[1]
        distance = math.sqrt(dx*dx + dy*dy)
        goal_theta = goal_pose[2] if len(goal_pose) > 2 else 0.0

        # Determine number of steps needed
        num_steps = int(distance / self.step_length) + 1

        # Generate footsteps along the path
        for i in range(1, num_steps + 1):
            ratio = i / num_steps
            x = start_pose[0] + ratio * dx
            y = start_pose[1] + ratio * dy
            theta = start_pose[2] + ratio * (goal_theta - start_pose[2])

            # Add slight variations for natural walking
            step = [x, y, theta]
            footsteps.append(step)

        # Add final goal step
        footsteps.append(goal_pose)

        return footsteps

    def plan_bipedal_sequence(self, footsteps):
        """
        Convert footsteps to alternating left/right foot sequence
        """
        sequence = []

        if not footsteps:
            return sequence

        # Start with left foot
        left_support = True

        for i, step in enumerate(footsteps):
            if left_support:
                # Left foot moves, right stays in place
                sequence.append({
                    'step_type': 'left',
                    'position': step[:2],
                    'orientation': step[2] if len(step) > 2 else 0.0,
                    'step_number': i
                })
                left_support = False
            else:
                # Right foot moves, left stays in place
                sequence.append({
                    'step_type': 'right',
                    'position': step[:2],
                    'orientation': step[2] if len(step) > 2 else 0.0,
                    'step_number': i
                })
                left_support = True

        return sequence

    def check_stability(self, footsteps, com_trajectory=None):
        """
        Check if the footstep sequence maintains stability
        """
        if len(footsteps) < 2:
            return True

        # Calculate support polygon for each step
        for i in range(len(footsteps) - 1):
            left_pos = self.get_left_foot_position(footsteps[i])
            right_pos = self.get_right_foot_position(footsteps[i])

            # Calculate support polygon (simplified as line between feet)
            support_polygon = [left_pos, right_pos]

            # Check if next step is within support polygon
            next_pos = footsteps[i + 1][:2]

            if not self.is_in_support_polygon(next_pos, support_polygon):
                return False

        return True

    def get_left_foot_position(self, step):
        """Get left foot position based on step"""
        x, y, theta = step[0], step[1], step[2]
        # Offset for left foot (simplified)
        offset_x = -self.foot_separation/2 * math.sin(theta)
        offset_y = self.foot_separation/2 * math.cos(theta)
        return [x + offset_x, y + offset_y]

    def get_right_foot_position(self, step):
        """Get right foot position based on step"""
        x, y, theta = step[0], step[1], step[2]
        # Offset for right foot (simplified)
        offset_x = self.foot_separation/2 * math.sin(theta)
        offset_y = -self.foot_separation/2 * math.cos(theta)
        return [x + offset_x, y + offset_y]

    def is_in_support_polygon(self, point, polygon):
        """Check if point is in support polygon (simplified)"""
        # Simplified check - in reality, this would be more complex
        if len(polygon) < 2:
            return False

        # Calculate distance to closest point in polygon
        min_dist = float('inf')
        for p in polygon:
            dist = euclidean(point, p)
            if dist < min_dist:
                min_dist = dist

        # Consider stable if within step distance
        return min_dist <= self.step_length

class AdvancedFootstepPlanner(FootstepPlanner):
    """Advanced footstep planner with stability and terrain considerations"""

    def __init__(self, step_length=0.3, step_width=0.2):
        super().__init__(step_length, step_width)
        self.max_step_up = 0.1
        self.max_step_down = 0.15
        self.max_com_velocity = 0.5  # m/s

    def plan_with_terrain(self, start_pose, goal_pose, height_map, obstacles=None):
        """
        Plan footsteps considering terrain elevation and obstacles
        """
        # Use A* or RRT for terrain-aware planning
        path = self.terrain_aware_search(start_pose, goal_pose, height_map, obstacles)

        # Smooth the path and generate footsteps
        footsteps = self.smooth_path_to_footsteps(path, height_map)

        return footsteps

    def terrain_aware_search(self, start, goal, height_map, obstacles):
        """Search for valid path considering terrain constraints"""
        # Simplified implementation - in practice, use proper path planning
        path = [start, goal]
        return path

    def smooth_path_to_footsteps(self, path, height_map):
        """Convert path to stable footsteps considering terrain"""
        footsteps = []

        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]

            # Calculate intermediate steps based on terrain
            step = self.calculate_terrain_aware_step(start, end, height_map)
            footsteps.append(step)

        return footsteps

    def calculate_terrain_aware_step(self, start, end, height_map):
        """Calculate step considering terrain constraints"""
        # Check elevation change
        start_height = self.get_terrain_height(start[:2], height_map)
        end_height = self.get_terrain_height(end[:2], height_map)

        height_diff = abs(end_height - start_height)

        if height_diff > self.max_step_up:
            # Need to find alternative path or stop
            # For now, return the direct step
            pass

        return end

    def get_terrain_height(self, position, height_map):
        """Get terrain height at position"""
        # Simplified - in practice, interpolate from height map
        return 0.0  # Default flat terrain

def visualize_footsteps(footsteps, sequence=None):
    """Visualize planned footsteps"""
    if not footsteps:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Extract x, y coordinates
    x_coords = [step[0] for step in footsteps]
    y_coords = [step[1] for step in footsteps]

    ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Planned Path')
    ax.scatter(x_coords, y_coords, c='red', s=50, zorder=5, label='Footsteps')

    # Mark start and end
    ax.scatter(x_coords[0], y_coords[0], c='green', s=100, zorder=6, label='Start')
    ax.scatter(x_coords[-1], y_coords[-1], c='red', s=100, zorder=6, label='Goal')

    # Draw foot shapes if sequence is provided
    if sequence:
        for step in sequence:
            x, y = step['position'][0], step['position'][1]
            theta = step['orientation']

            # Draw simple foot shape
            foot_length = 0.15
            foot_width = 0.07

            # Calculate foot corners
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)

            corners_x = []
            corners_y = []

            for dx, dy in [(-foot_length/2, -foot_width/2),
                           (foot_length/2, -foot_width/2),
                           (foot_length/2, foot_width/2),
                           (-foot_length/2, foot_width/2),
                           (-foot_length/2, -foot_width/2)]:
                x_corner = x + dx * cos_theta - dy * sin_theta
                y_corner = y + dx * sin_theta + dy * cos_theta
                corners_x.append(x_corner)
                corners_y.append(y_corner)

            color = 'blue' if step['step_type'] == 'left' else 'orange'
            ax.plot(corners_x, corners_y, color=color, linewidth=1)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Footstep Planning for Humanoid Navigation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

    plt.tight_layout()
    plt.show()

# Example usage
def example_footstep_planning():
    planner = AdvancedFootstepPlanner()

    # Define start and goal
    start_pose = [0.0, 0.0, 0.0]  # x, y, theta
    goal_pose = [3.0, 2.0, math.pi/4]

    # Plan footsteps
    footsteps = planner.plan_footsteps(start_pose, goal_pose)
    sequence = planner.plan_bipedal_sequence(footsteps)

    # Check stability
    is_stable = planner.check_stability(footsteps)
    print(f"Footstep sequence is stable: {is_stable}")

    # Visualize
    visualize_footsteps(footsteps, sequence)

    return footsteps, sequence
```

### Whole-Body Motion Planning Integration

```python
# python/whole_body_motion_planning.py
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import casadi as cs

class WholeBodyMotionPlanner:
    def __init__(self):
        self.robot_params = {
            'height': 1.5,  # m
            'weight': 60.0, # kg
            'com_height': 0.8,  # m (height of center of mass)
            'foot_size': [0.25, 0.15]  # length, width
        }

        self.motion_constraints = {
            'max_velocity': 0.5,  # m/s
            'max_angular_velocity': 0.5,  # rad/s
            'max_acceleration': 1.0,  # m/s^2
            'max_angular_acceleration': 1.0  # rad/s^2
        }

    def plan_whole_body_motion(self, footsteps, start_state, goal_state):
        """
        Plan whole-body motion to execute footsteps while maintaining balance
        """
        # Optimize center of mass trajectory to match footsteps
        com_trajectory = self.optimize_com_trajectory(footsteps, start_state, goal_state)

        # Generate joint trajectories
        joint_trajectories = self.inverse_kinematics(com_trajectory, footsteps)

        # Generate balance control commands
        balance_commands = self.generate_balance_control(com_trajectory)

        return {
            'com_trajectory': com_trajectory,
            'joint_trajectories': joint_trajectories,
            'balance_commands': balance_commands
        }

    def optimize_com_trajectory(self, footsteps, start_state, goal_state):
        """
        Optimize center of mass trajectory for stable locomotion
        """
        # Use preview control or other methods to generate CoM trajectory
        # that ensures ZMP (Zero Moment Point) stays within support polygon

        # Simplified approach: generate CoM trajectory that follows footsteps
        # with appropriate smoothing for balance
        com_trajectory = []

        for i, step in enumerate(footsteps):
            # Calculate desired CoM position based on step location
            # and support polygon
            com_x = step[0]
            com_y = step[1]
            com_z = self.robot_params['com_height']  # Keep CoM at constant height

            # Add time parameter
            time = i * 0.5  # Assume 0.5s per step

            com_trajectory.append([time, com_x, com_y, com_z])

        return com_trajectory

    def inverse_kinematics(self, com_trajectory, footsteps):
        """
        Calculate joint angles to achieve desired CoM position and foot placement
        """
        joint_trajectories = []

        for t, com_pos in enumerate(com_trajectory):
            # Calculate required joint angles using inverse kinematics
            # This is a simplified approach - real implementation would use
            # full kinematic model of the humanoid
            joint_angles = self.calculate_joint_angles(com_pos, footsteps, t)
            joint_trajectories.append(joint_angles)

        return joint_trajectories

    def calculate_joint_angles(self, com_pos, footsteps, time_idx):
        """
        Calculate joint angles for given CoM position
        """
        # Simplified joint angle calculation
        # In reality, this would solve the full inverse kinematics problem
        joints = {
            'left_hip': [0.0, 0.0, 0.0],  # [roll, pitch, yaw]
            'left_knee': [0.0],            # [flexion]
            'left_ankle': [0.0, 0.0],      # [pitch, roll]
            'right_hip': [0.0, 0.0, 0.0],
            'right_knee': [0.0],
            'right_ankle': [0.0, 0.0],
            'left_shoulder': [0.0, 0.0, 0.0],
            'left_elbow': [0.0],
            'right_shoulder': [0.0, 0.0, 0.0],
            'right_elbow': [0.0],
            'torso': [0.0, 0.0, 0.0]
        }

        return joints

    def generate_balance_control(self, com_trajectory):
        """
        Generate balance control commands to maintain stability
        """
        balance_commands = []

        for i in range(1, len(com_trajectory)):
            current_com = np.array(com_trajectory[i][1:4])  # x, y, z
            prev_com = np.array(com_trajectory[i-1][1:4])

            # Calculate CoM velocity
            dt = com_trajectory[i][0] - com_trajectory[i-1][0]
            if dt > 0:
                com_velocity = (current_com - prev_com) / dt
            else:
                com_velocity = np.array([0.0, 0.0, 0.0])

            # Generate balance command based on CoM state
            balance_cmd = self.calculate_balance_command(current_com, com_velocity)
            balance_commands.append(balance_cmd)

        return balance_commands

    def calculate_balance_command(self, com_pos, com_vel):
        """
        Calculate balance command using inverted pendulum model
        """
        # Simplified balance control using linear inverted pendulum model
        # (LIPM) - in reality, this would be more complex

        # Desired CoM position (based on support polygon)
        desired_com_x = com_pos[0]  # Simplified
        desired_com_y = com_pos[1]

        # Calculate error
        x_error = desired_com_x - com_pos[0]
        y_error = desired_com_y - com_pos[1]

        # Simple PD control
        kp = 10.0  # Proportional gain
        kd = 2.0   # Derivative gain (velocity feedback)

        x_control = kp * x_error - kd * com_vel[0]
        y_control = kp * y_error - kd * com_vel[1]

        return [x_control, y_control]

class PreviewController:
    """
    Implement preview control for humanoid balance during locomotion
    """
    def __init__(self, zmp_delay=0.05, preview_window=1.0):
        self.zmp_delay = zmp_delay
        self.preview_window = preview_window
        self.gravity = 9.81
        self.com_height = 0.8  # m

        # Calculate omega for LIPM
        self.omega = math.sqrt(self.gravity / self.com_height)

    def calculate_com_reference(self, zmp_trajectory):
        """
        Calculate CoM reference trajectory from ZMP trajectory using preview control
        """
        # Implement preview control algorithm
        # This ensures ZMP follows desired trajectory while maintaining stability

        com_reference = []

        for i, (t, zmp_x, zmp_y) in enumerate(zmp_trajectory):
            # Calculate reference CoM position using preview control
            # This is a simplified implementation
            com_x = zmp_x  # Simplified - in reality, would use full preview control
            com_y = zmp_y
            com_z = self.com_height

            com_reference.append([t, com_x, com_y, com_z])

        return com_reference

    def generate_zmp_trajectory(self, footsteps):
        """
        Generate ZMP trajectory from footsteps
        """
        zmp_trajectory = []

        for i, step in enumerate(footsteps):
            # Calculate ZMP based on foot placement and timing
            time = i * 0.5  # Assume 0.5s per step
            zmp_x = step[0]
            zmp_y = step[1]

            zmp_trajectory.append([time, zmp_x, zmp_y])

        return zmp_trajectory

def create_balanced_locomotion_plan(footsteps, start_state, goal_state):
    """
    Create a complete locomotion plan with balance considerations
    """
    # Initialize planners
    wb_planner = WholeBodyMotionPlanner()
    preview_ctrl = PreviewController()

    # Generate ZMP trajectory from footsteps
    zmp_trajectory = preview_ctrl.generate_zmp_trajectory(footsteps)

    # Calculate CoM reference using preview control
    com_reference = preview_ctrl.calculate_com_reference(zmp_trajectory)

    # Plan whole body motion
    motion_plan = wb_planner.plan_whole_body_motion(
        footsteps, start_state, goal_state
    )

    return {
        'footsteps': footsteps,
        'zmp_trajectory': zmp_trajectory,
        'com_reference': com_reference,
        'whole_body_plan': motion_plan
    }
```

## Dynamic Balance Control

### Balance Control Algorithms

```python
# python/balance_control.py
import numpy as np
import math
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from scipy import signal

class BalanceController:
    def __init__(self, robot_height=0.8, control_frequency=100):
        self.com_height = robot_height  # Height of center of mass
        self.control_frequency = control_frequency
        self.gravity = 9.81

        # State: [x, y, x_dot, y_dot] - CoM position and velocity
        self.state = np.zeros(4)

        # Calculate LQR gains for inverted pendulum
        self.A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [self.gravity/self.com_height, 0, 0, 0],
            [0, self.gravity/self.com_height, 0, 0]
        ])

        # Control input matrix (how control affects acceleration)
        self.B = np.array([
            [0, 0],
            [0, 0],
            [-self.gravity/self.com_height, 0],
            [0, -self.gravity/self.com_height]
        ])

        # State cost matrix Q (penalizes state deviations)
        self.Q = np.diag([100, 100, 10, 10])  # [x_pos, y_pos, x_vel, y_vel]

        # Control cost matrix R (penalizes control effort)
        self.R = np.diag([1, 1])  # [x_control, y_control]

        # Calculate LQR gain matrix
        self.K = self.calculate_lqr_gain()

    def calculate_lqr_gain(self):
        """Calculate LQR gain matrix for inverted pendulum control"""
        # Solve continuous-time algebraic Riccati equation
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # LQR gain: u = -K*x
        K = np.linalg.inv(self.R) @ self.B.T @ P

        return K

    def update_balance_control(self, current_state, desired_state):
        """
        Update balance control based on current and desired state
        current_state: [com_x, com_y, com_x_dot, com_y_dot]
        desired_state: [des_x, des_y, des_x_dot, des_y_dot]
        """
        # Calculate state error
        state_error = desired_state - current_state

        # Apply LQR control law: u = -K*(x - x_desired)
        control_output = -self.K @ state_error

        # Control output represents desired ZMP displacement
        zmp_dx, zmp_dy = control_output

        return zmp_dx, zmp_dy

    def integrate_dynamics(self, state, t, zmp_x, zmp_y):
        """
        Integrate inverted pendulum dynamics
        state: [x, y, x_dot, y_dot]
        """
        x, y, x_dot, y_dot = state

        # Inverted pendulum dynamics
        # x_ddot = g/h * (x - zmp_x)
        # y_ddot = g/h * (y - zmp_y)
        x_ddot = self.gravity/self.com_height * (x - zmp_x)
        y_ddot = self.gravity/self.com_height * (y - zmp_y)

        return [x_dot, y_dot, x_ddot, y_ddot]

    def simulate_balance(self, initial_state, zmp_trajectory, dt=0.01):
        """
        Simulate balance control over time
        """
        states = [initial_state]
        times = [0.0]

        current_state = initial_state.copy()

        for t in np.arange(0, len(zmp_trajectory)*dt, dt):
            # Get desired ZMP for current time
            idx = min(int(t/dt), len(zmp_trajectory)-1)
            desired_zmp_x, desired_zmp_y = zmp_trajectory[idx]

            # Calculate control (simplified - in reality, would use full state feedback)
            state_error = current_state - np.array([desired_zmp_x, desired_zmp_y, 0, 0])
            control = -self.K @ state_error

            # Apply control to ZMP
            actual_zmp_x = desired_zmp_x + control[0]
            actual_zmp_y = desired_zmp_y + control[1]

            # Integrate dynamics
            derivatives = self.integrate_dynamics(
                current_state, t, actual_zmp_x, actual_zmp_y
            )

            # Update state (Euler integration)
            new_state = current_state + np.array(derivatives) * dt
            current_state = new_state

            states.append(new_state.copy())
            times.append(t + dt)

        return np.array(times), np.array(states)

class CapturePointController:
    """
    Capture Point based balance control for humanoid robots
    """
    def __init__(self, com_height=0.8, control_frequency=200):
        self.com_height = com_height
        self.control_frequency = control_frequency
        self.gravity = 9.81
        self.omega = math.sqrt(self.gravity / self.com_height)

    def calculate_capture_point(self, com_pos, com_vel):
        """
        Calculate capture point from CoM position and velocity
        Capture point = CoM position + CoM velocity / omega
        """
        cp_x = com_pos[0] + com_vel[0] / self.omega
        cp_y = com_pos[1] + com_vel[1] / self.omega

        return np.array([cp_x, cp_y])

    def calculate_foot_placement(self, capture_point, current_foot_pos):
        """
        Calculate required foot placement to capture the current state
        """
        # For simplicity, move foot towards capture point
        # In reality, this would consider step constraints and timing
        step_vector = capture_point - current_foot_pos
        max_step_length = 0.3  # Maximum step length

        if np.linalg.norm(step_vector) > max_step_length:
            step_vector = step_vector / np.linalg.norm(step_vector) * max_step_length

        new_foot_pos = current_foot_pos + step_vector

        return new_foot_pos

    def balance_control_step(self, com_state, support_foot_pos):
        """
        Perform one step of capture point based balance control
        com_state: [x, y, z, x_dot, y_dot, z_dot]
        support_foot_pos: [x, y] of current support foot
        """
        com_pos = com_state[:3]
        com_vel = com_state[3:]

        # Calculate capture point
        cp = self.calculate_capture_point(com_pos[:2], com_vel[:2])

        # Determine if step is needed
        foot_to_cp = np.linalg.norm(cp - support_foot_pos)
        stability_threshold = 0.1  # Start planning step when CP is 10cm from foot

        step_needed = foot_to_cp > stability_threshold

        return {
            'capture_point': cp,
            'distance_to_foot': foot_to_cp,
            'step_needed': step_needed,
            'target_foot_placement': self.calculate_foot_placement(cp, support_foot_pos) if step_needed else support_foot_pos
        }

class PendulumController:
    """
    Linear Inverted Pendulum Mode (LIPM) controller
    """
    def __init__(self, com_height=0.8):
        self.com_height = com_height
        self.gravity = 9.81
        self.omega = math.sqrt(self.gravity / self.com_height)

        # Discrete time model parameters
        self.dt = 0.005  # 200Hz control rate

    def discrete_dynamics(self, x_k, zmp_k):
        """
        Discrete time inverted pendulum dynamics
        x_k = [com_x, com_x_dot] at time k
        zmp_k = desired ZMP at time k
        """
        # State transition matrix for discrete LIPM
        A_d = np.array([
            [np.cosh(self.omega * self.dt), (1/self.omega) * np.sinh(self.omega * self.dt)],
            [self.omega * np.sinh(self.omega * self.dt), np.cosh(self.omega * self.dt)]
        ])

        # Input matrix
        B_d = np.array([
            [1 - np.cosh(self.omega * self.dt)],
            [-self.omega * np.sinh(self.omega * self.dt)]
        ])

        # Next state
        x_k1 = A_d @ x_k + B_d * zmp_k

        return x_k1

    def mpc_balance_control(self, current_state, reference_trajectory, horizon=20):
        """
        Model Predictive Control for balance using LIPM
        """
        # Simplified MPC implementation
        # In reality, this would solve a constrained optimization problem

        predicted_states = []
        control_inputs = []

        current_x = current_state.copy()

        for k in range(horizon):
            # Get reference for this step
            if k < len(reference_trajectory):
                ref_state = reference_trajectory[k]
            else:
                ref_state = reference_trajectory[-1]  # Hold last reference

            # Simple control law (in reality, would solve MPC optimization)
            zmp_ref = ref_state[0]  # Reference ZMP
            current_zmp = current_x[0]  # Current CoM position

            # Proportional control to track reference
            zmp_cmd = zmp_ref + 0.1 * (current_zmp - ref_state[0])

            # Apply dynamics
            next_x = self.discrete_dynamics(current_x, zmp_cmd)

            predicted_states.append(next_x.copy())
            control_inputs.append(zmp_cmd)

            current_x = next_x

        # Return first control input
        return control_inputs[0] if control_inputs else 0.0

def demonstrate_balance_control():
    """
    Demonstrate different balance control approaches
    """
    print("Demonstrating Balance Control Approaches")

    # Initialize controllers
    lqr_controller = BalanceController()
    cp_controller = CapturePointController()
    lipm_controller = PendulumController()

    # Example CoM state [x, y, x_dot, y_dot]
    com_state = np.array([0.05, 0.02, 0.1, -0.05])  # Slightly perturbed

    # Example desired state
    desired_state = np.array([0.0, 0.0, 0.0, 0.0])  # At equilibrium

    # Calculate LQR control
    zmp_dx, zmp_dy = lqr_controller.update_balance_control(com_state, desired_state)
    print(f"LQR Control - ZMP adjustment: ({zmp_dx:.3f}, {zmp_dy:.3f})")

    # Example CoM state for capture point controller [x, y, z, x_dot, y_dot, z_dot]
    full_com_state = np.array([0.05, 0.02, 0.8, 0.1, -0.05, 0.0])
    support_foot = np.array([0.0, 0.0])

    cp_result = cp_controller.balance_control_step(full_com_state, support_foot)
    print(f"Capture Point: ({cp_result['capture_point'][0]:.3f}, {cp_result['capture_point'][1]:.3f})")
    print(f"Step needed: {cp_result['step_needed']}")

    return lqr_controller, cp_controller, lipm_controller
```

## Nav2 Behavior Trees for Humanoid Navigation

### Custom Behavior Trees

```python
# python/humanoid_behavior_trees.py
import py_trees
import py_trees_ros
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
import time

class HumanoidNavigateToPoseAction(py_trees_ros.actions.ActionClient):
    """
    Custom action client for humanoid navigation to pose
    """
    def __init__(self, name, action_type, action_name, goal):
        super().__init__(name=name, action_type=action_type, action_name=action_name, goal=goal)
        self.feedback_message = "initialised"

    def update(self):
        # Check if action server is available
        if not self.action_client.server_is_ready():
            return py_trees.Status.RUNNING

        # Send goal if not already sent
        if self.sent_goal is False:
            self.send_goal()
            return py_trees.Status.RUNNING

        # Check goal status
        if self.goal_handle.status == GoalStatus.STATUS_SUCCEEDED:
            self.feedback_message = "arrived at goal"
            return py_trees.Status.SUCCESS
        elif self.goal_handle.status == GoalStatus.STATUS_EXECUTING:
            self.feedback_message = "moving to goal"
            return py_trees.Status.RUNNING
        else:
            self.feedback_message = "navigation failed"
            return py_trees.Status.FAILURE

class CheckFootSupport(py_trees.behaviour.Behaviour):
    """
    Check if the robot has proper foot support before navigation
    """
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, **kwargs):
        # Initialize any required resources
        pass

    def update(self):
        # Check if robot has stable foot support
        # This would interface with robot state
        has_support = self.check_robot_support()

        if has_support:
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

    def check_robot_support(self):
        # Simulate checking for foot support
        # In reality, this would check robot's balance state
        return True  # Assume stable for example

class PlanFootsteps(py_trees.behaviour.Behaviour):
    """
    Plan safe footsteps to the goal
    """
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, **kwargs):
        # Initialize footstep planner
        self.footstep_planner = FootstepPlanner()

    def update(self):
        # Get goal from blackboard
        goal = self.blackboard.get("navigation_goal")

        if goal is None:
            return py_trees.Status.FAILURE

        # Plan footsteps
        start_pose = [0.0, 0.0, 0.0]  # Current pose
        footsteps = self.footstep_planner.plan_footsteps(start_pose, goal)

        if footsteps:
            self.blackboard.set("footsteps", footsteps)
            self.feedback_message = f"Planned {len(footsteps)} footsteps"
            return py_trees.Status.SUCCESS
        else:
            return py_trees.Status.FAILURE

class ExecuteFootsteps(py_trees.behaviour.Behaviour):
    """
    Execute planned footsteps with balance control
    """
    def __init__(self, name):
        super().__init__(name)
        self.blackboard = py_trees.blackboard.Blackboard()

    def setup(self, **kwargs):
        # Initialize balance controller
        self.balance_controller = BalanceController()

    def update(self):
        footsteps = self.blackboard.get("footsteps")

        if not footsteps:
            return py_trees.Status.FAILURE

        # Execute footsteps one by one
        current_step_idx = self.blackboard.get("current_step", 0)

        if current_step_idx >= len(footsteps):
            # All footsteps executed
            return py_trees.Status.SUCCESS

        # Execute current step with balance control
        success = self.execute_single_step(footsteps[current_step_idx])

        if success:
            self.blackboard.set("current_step", current_step_idx + 1)
            self.feedback_message = f"Completed step {current_step_idx + 1}"
            return py_trees.Status.RUNNING  # Continue to next step
        else:
            return py_trees.Status.FAILURE

    def execute_single_step(self, step):
        # Simulate executing a single step with balance control
        # In reality, this would interface with robot's walking controller
        time.sleep(0.1)  # Simulate step execution time
        return True  # Assume success for example

class HumanoidNavigationSelector(py_trees.composites.Selector):
    """
    Selector for humanoid navigation behaviors
    """
    def __init__(self, name):
        super().__init__(name)

        # Add child behaviors
        self.add_child(CheckFootSupport("CheckSupport"))
        self.add_child(PlanFootsteps("PlanFootsteps"))
        self.add_child(ExecuteFootsteps("ExecuteFootsteps"))

def create_humanoid_navigation_tree():
    """
    Create a behavior tree for humanoid navigation
    """
    # Main root
    root = py_trees.composites.Sequence(name="HumanoidNavigation")

    # Add selector for navigation behaviors
    nav_selector = HumanoidNavigationSelector("NavigationSelector")
    root.add_child(nav_selector)

    # Add goal setting (in practice, this would come from external source)
    set_goal = py_trees.behaviours.Success("SetGoal")
    root.insert_child(set_goal, index=0)

    return root

class BehaviorTreeManager(Node):
    """
    Manage the behavior tree execution
    """
    def __init__(self):
        super().__init__('behavior_tree_manager')

        # Create the behavior tree
        self.tree = create_humanoid_navigation_tree()

        # Setup tree visitor for debugging
        self.snapshot_visitor = py_trees.visitors.SnapshotVisitor()

        # Timer for tree ticking
        self.timer = self.create_timer(0.1, self.tick_tree)

        self.get_logger().info("Behavior Tree Manager initialized")

    def tick_tree(self):
        """
        Tick the behavior tree
        """
        # Tick the tree
        self.tree.tick_once()

        # Visit the tree to get feedback
        self.tree.visit(self.snapshot_visitor)

        # Print tree status
        print(py_trees.display.unicode_tree(
            root=self.tree,
            visited=self.snapshot_visitor.visited,
            previously_visited=self.snapshot_visitor.previously_visited
        ))

        # Check if tree has reached a conclusion
        if self.tree.status == py_trees.common.Status.SUCCESS:
            self.get_logger().info("Navigation task completed successfully")
        elif self.tree.status == py_trees.common.Status.FAILURE:
            self.get_logger().error("Navigation task failed")

        # Reset tree if completed
        if self.tree.status in [py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE]:
            self.tree.tip().stop(py_trees.common.Status.INVALID)
            self.tree.reset()

def main(args=None):
    rclpy.init(args=args)

    # Initialize behavior tree manager
    bt_manager = BehaviorTreeManager()

    try:
        rclpy.spin(bt_manager)
    except KeyboardInterrupt:
        pass
    finally:
        bt_manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization and Real-Time Considerations

### Real-Time Navigation Optimization

```python
# python/navigation_optimization.py
import numpy as np
import time
import threading
from collections import deque
import multiprocessing as mp
from numba import jit, cuda
import ctypes

class RealTimeNavigationOptimizer:
    """
    Optimize navigation algorithms for real-time humanoid locomotion
    """
    def __init__(self):
        self.control_frequency = 200  # Hz
        self.planning_frequency = 10  # Hz
        self.safety_frequency = 100   # Hz

        # Real-time constraints
        self.max_control_time = 1.0 / self.control_frequency  # seconds
        self.max_planning_time = 1.0 / self.planning_frequency
        self.max_safety_time = 1.0 / self.safety_frequency

        # Threading for parallel processing
        self.control_thread = None
        self.planning_thread = None
        self.safety_thread = None

        # Data buffers
        self.sensor_data = deque(maxlen=10)
        self.trajectory_buffer = deque(maxlen=5)
        self.control_commands = deque(maxlen=5)

    @jit(nopython=True)
    def fast_distance_calculation(self, p1, p2):
        """Fast distance calculation using Numba"""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return np.sqrt(dx*dx + dy*dy)

    @jit(nopython=True)
    def fast_vector_operations(self, v1, v2):
        """Fast vector operations using Numba"""
        # Add two vectors
        result = np.empty(2)
        result[0] = v1[0] + v2[0]
        result[1] = v1[1] + v2[1]
        return result

    def run_control_loop(self):
        """Run the real-time control loop"""
        rate = 1.0 / self.control_frequency

        while True:
            start_time = time.time()

            try:
                # Execute control algorithm
                control_cmd = self.execute_control_algorithm()

                # Add to command buffer
                self.control_commands.append(control_cmd)

                # Maintain real-time constraint
                elapsed = time.time() - start_time
                sleep_time = max(0, rate - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.get_logger().error(f"Control loop error: {e}")
                time.sleep(rate)

    def execute_control_algorithm(self):
        """Execute the main control algorithm"""
        # Simplified control algorithm
        # In reality, this would implement the full control law

        # Get latest sensor data
        if self.sensor_data:
            latest_data = self.sensor_data[-1]
        else:
            return np.zeros(6)  # Default command

        # Calculate control output (simplified)
        control_output = np.zeros(6)  # [x_vel, y_vel, theta_vel, com_x, com_y, com_z]

        # Apply control law
        # This would implement the balance control algorithm
        control_output[0] = latest_data.get('desired_x_vel', 0.0)
        control_output[1] = latest_data.get('desired_y_vel', 0.0)
        control_output[2] = latest_data.get('desired_theta_vel', 0.0)

        return control_output

    def run_planning_loop(self):
        """Run the path planning loop"""
        rate = 1.0 / self.planning_frequency

        while True:
            start_time = time.time()

            try:
                # Execute planning algorithm
                new_plan = self.execute_planning_algorithm()

                # Add to trajectory buffer
                if new_plan is not None:
                    self.trajectory_buffer.append(new_plan)

                # Maintain real-time constraint
                elapsed = time.time() - start_time
                sleep_time = max(0, rate - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.get_logger().error(f"Planning loop error: {e}")
                time.sleep(rate)

    def execute_planning_algorithm(self):
        """Execute the path planning algorithm"""
        # Simplified planning algorithm
        # In reality, this would implement A*, RRT, or other planners

        # Check for new goal
        goal = self.get_latest_goal()
        if goal is None:
            return None

        # Get current state
        current_state = self.get_current_state()

        # Plan path (simplified)
        path = self.plan_path_fast(current_state, goal)

        return path

    def plan_path_fast(self, start, goal):
        """Fast path planning (simplified)"""
        # Use a fast but potentially suboptimal planner
        # For real-time applications, prioritize speed over optimality

        # Simplified straight-line path with obstacle avoidance
        path = [start, goal]
        return path

    def run_safety_loop(self):
        """Run the safety monitoring loop"""
        rate = 1.0 / self.safety_frequency

        while True:
            start_time = time.time()

            try:
                # Check safety conditions
                safety_status = self.check_safety_conditions()

                # Handle safety violations
                if not safety_status['is_safe']:
                    self.emergency_stop()

                # Maintain real-time constraint
                elapsed = time.time() - start_time
                sleep_time = max(0, rate - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                self.get_logger().error(f"Safety loop error: {e}")
                self.emergency_stop()
                time.sleep(rate)

    def check_safety_conditions(self):
        """Check various safety conditions"""
        safety_status = {
            'is_safe': True,
            'balance_safe': True,
            'collision_safe': True,
            'hardware_safe': True
        }

        # Check balance
        balance_ok = self.check_balance_safety()
        safety_status['balance_safe'] = balance_ok
        safety_status['is_safe'] &= balance_ok

        # Check for collisions
        collision_ok = self.check_collision_safety()
        safety_status['collision_safe'] = collision_ok
        safety_status['is_safe'] &= collision_ok

        # Check hardware status
        hardware_ok = self.check_hardware_safety()
        safety_status['hardware_safe'] = hardware_ok
        safety_status['is_safe'] &= hardware_ok

        return safety_status

    def check_balance_safety(self):
        """Check if robot is in safe balance state"""
        # Simplified balance check
        # In reality, this would check ZMP, CoM position, etc.
        return True

    def check_collision_safety(self):
        """Check for potential collisions"""
        # Simplified collision check
        # In reality, this would process sensor data
        return True

    def check_hardware_safety(self):
        """Check hardware status"""
        # Simplified hardware check
        # In reality, this would check joint limits, temperatures, etc.
        return True

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        self.get_logger().error("EMERGENCY STOP ACTIVATED")
        # Send zero commands to all actuators
        # This would interface with the robot's safety system

    def get_latest_goal(self):
        """Get the latest navigation goal"""
        # In reality, this would get goal from ROS topics
        return None

    def get_current_state(self):
        """Get the current robot state"""
        # In reality, this would get state from odometry, IMU, etc.
        return [0.0, 0.0, 0.0]

    def start_real_time_loops(self):
        """Start all real-time loops in separate threads"""
        # Start control loop
        self.control_thread = threading.Thread(target=self.run_control_loop, daemon=True)
        self.control_thread.start()

        # Start planning loop
        self.planning_thread = threading.Thread(target=self.run_planning_loop, daemon=True)
        self.planning_thread.start()

        # Start safety loop
        self.safety_thread = threading.Thread(target=self.run_safety_loop, daemon=True)
        self.safety_thread.start()

class MultiProcessNavigation:
    """
    Multi-process navigation system for better real-time performance
    """
    def __init__(self):
        self.processes = []
        self.shared_memory = {}

    def create_navigation_process(self):
        """Create a dedicated process for navigation"""
        # This would create a process with real-time priority
        # and dedicated CPU core if possible
        pass

    def setup_shared_memory(self):
        """Setup shared memory for inter-process communication"""
        # Use multiprocessing shared memory for low-latency communication
        pass

def optimize_for_real_time():
    """
    Apply various optimizations for real-time performance
    """
    optimizer = RealTimeNavigationOptimizer()

    # Apply Numba optimizations
    # The @jit decorators above will compile functions to machine code

    # Start real-time loops
    optimizer.start_real_time_loops()

    # Monitor performance
    monitor = PerformanceMonitor()
    monitor.start_monitoring()

    return optimizer

class PerformanceMonitor:
    """
    Monitor navigation system performance
    """
    def __init__(self):
        self.metrics = {
            'control_loop_times': deque(maxlen=1000),
            'planning_times': deque(maxlen=100),
            'loop_rates': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000)
        }
        self.start_time = time.time()

    def start_monitoring(self):
        """Start performance monitoring"""
        import psutil
        import threading

        def monitor_loop():
            while True:
                # Record current time
                current_time = time.time()

                # Monitor memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics['memory_usage'].append(memory_mb)

                # Calculate loop rate
                elapsed = current_time - self.start_time
                if elapsed > 0:
                    rate = len(self.metrics['loop_rates']) / elapsed
                    self.metrics['loop_rates'].append(rate)

                time.sleep(1.0)  # Update every second

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def get_performance_report(self):
        """Get current performance metrics"""
        if not self.metrics['loop_rates']:
            return "No data available"

        avg_rate = sum(self.metrics['loop_rates']) / len(self.metrics['loop_rates'])
        max_memory = max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0

        report = f"""
        Performance Report:
        - Average Loop Rate: {avg_rate:.2f} Hz
        - Max Memory Usage: {max_memory:.2f} MB
        - Control Loop Samples: {len(self.metrics['control_loop_times'])}
        - Planning Samples: {len(self.metrics['planning_times'])}
        """

        return report

# Example usage
def run_optimized_navigation():
    """
    Run the optimized navigation system
    """
    print("Starting optimized humanoid navigation system...")

    # Initialize optimizer
    nav_optimizer = optimize_for_real_time()

    # Run for a while
    time.sleep(10)

    # Print performance report
    monitor = PerformanceMonitor()
    print(monitor.get_performance_report())

    return nav_optimizer
```

## Integration with Isaac Sim and ROS 2

### Isaac Sim Navigation Integration

```python
# python/isaac_sim_nav_integration.py
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import math
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

class IsaacSimNavigationInterface:
    """
    Interface between Isaac Sim and ROS 2 for humanoid navigation
    """
    def __init__(self, robot_name="humanoid_robot", stage_units_in_meters=1.0):
        # Isaac Sim components
        self.world = World(stage_units_in_meters=stage_units_in_meters)
        self.robot = None
        self.robot_name = robot_name

        # ROS 2 components (will be initialized later)
        self.ros_node = None
        self.ros_initialized = False

        # Navigation state
        self.current_goal = None
        self.navigation_active = False
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta

    def setup_isaac_sim_environment(self):
        """Setup Isaac Sim environment with humanoid robot"""
        # Get assets root path
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets path")

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add humanoid robot (using a simple model for demonstration)
        robot_path = f"{assets_root_path}/Isaac/Robots/Franka/fr3.usd"
        add_reference_to_stage(usd_path=robot_path, prim_path="/World/Robot")

        # Initialize robot in Isaac Sim
        self.robot = self.world.scene.get_object("Robot")

        # Add obstacles to the environment
        self.add_obstacles()

        # Setup camera for visualization
        set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.0])

    def add_obstacles(self):
        """Add obstacles to the environment"""
        # Add some simple obstacles
        obstacles = [
            {"name": "box1", "position": [1.0, 0.0, 0.1], "size": [0.2, 0.2, 0.2]},
            {"name": "box2", "position": [0.0, 1.0, 0.1], "size": [0.2, 0.2, 0.2]},
            {"name": "box3", "position": [-1.0, -1.0, 0.1], "size": [0.2, 0.2, 0.2]},
        ]

        for obs in obstacles:
            DynamicCuboid(
                prim_path=f"/World/{obs['name']}",
                name=obs['name'],
                position=obs['position'],
                size=obs['size'],
                color=np.array([0.5, 0.5, 0.5])
            )

    def initialize_ros_interface(self):
        """Initialize ROS 2 interface"""
        if not self.ros_initialized:
            rclpy.init()
            self.ros_node = IsaacSimRosBridge()
            self.ros_initialized = True

    def run_navigation_simulation(self):
        """Run the navigation simulation loop"""
        # Reset the world
        self.world.reset()

        # Initialize ROS interface
        self.initialize_ros_interface()

        # Main simulation loop
        sim_step = 0
        while True:
            # Step Isaac Sim
            self.world.step(render=True)

            # Update robot pose
            self.update_robot_pose()

            # Process ROS messages
            if self.ros_node:
                try:
                    rclpy.spin_once(self.ros_node, timeout_sec=0)
                except KeyboardInterrupt:
                    break

            # Process navigation commands
            self.process_navigation_commands()

            # Log progress
            if sim_step % 100 == 0:
                print(f"Simulation step: {sim_step}, Robot pose: {self.robot_pose}")

            sim_step += 1

            # Limit simulation steps for demo
            if sim_step > 5000:  # Run for 5000 steps then stop
                break

    def update_robot_pose(self):
        """Update robot pose from Isaac Sim"""
        if self.robot:
            # Get current pose from Isaac Sim
            pose = self.robot.get_world_poses()
            if pose:
                positions, orientations = pose
                if len(positions) > 0:
                    pos = positions[0]
                    # Simplified: extract x, y position and approximate theta
                    self.robot_pose[0] = float(pos[0])
                    self.robot_pose[1] = float(pos[1])

                    # For theta, we'd need orientation information
                    # Simplified for this example
                    self.robot_pose[2] = 0.0  # Placeholder

    def process_navigation_commands(self):
        """Process navigation commands from ROS"""
        if self.current_goal is not None and self.navigation_active:
            # Calculate distance to goal
            dx = self.current_goal[0] - self.robot_pose[0]
            dy = self.current_goal[1] - self.robot_pose[1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Check if reached goal
            if distance < 0.2:  # 20cm threshold
                self.navigation_active = False
                if self.ros_node:
                    self.ros_node.publish_navigation_status("GOAL_REACHED")
            else:
                # Send movement command towards goal
                self.move_towards_goal(dx, dy)

    def move_towards_goal(self, dx, dy):
        """Send command to move robot towards goal"""
        if self.ros_node:
            # Calculate desired velocity
            speed = min(0.2, math.sqrt(dx*dx + dy*dy) * 2.0)  # Proportional control
            angle = math.atan2(dy, dx)

            # Create twist message
            twist_msg = Twist()
            twist_msg.linear.x = speed * math.cos(angle - self.robot_pose[2])
            twist_msg.angular.z = angle - self.robot_pose[2]

            # Publish command
            self.ros_node.publish_velocity_command(twist_msg)

class IsaacSimRosBridge(Node):
    """
    ROS 2 bridge node for Isaac Sim integration
    """
    def __init__(self):
        super().__init__('isaac_sim_nav_bridge')

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/move_base_simple/goal',
            self.goal_callback,
            10
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Timer for publishing sensor data
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

        self.get_logger().info("Isaac Sim ROS Bridge initialized")

    def goal_callback(self, msg):
        """Handle navigation goal from ROS"""
        goal = [msg.pose.position.x, msg.pose.position.y, 0.0]
        self.get_logger().info(f"Received navigation goal: {goal}")

        # This would be passed to the navigation system
        # For now, we'll just log it

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS"""
        self.get_logger().info(f"Received velocity command: {msg.linear.x}, {msg.angular.z}")

        # This would be sent to the robot in Isaac Sim
        # For now, we'll just log it

    def publish_sensor_data(self):
        """Publish sensor data from Isaac Sim"""
        # Publish odometry (simplified)
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Publish dummy data for demonstration
        odom_msg.pose.pose.position.x = 0.0
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.position.z = 0.0

        self.odom_pub.publish(odom_msg)

        # Publish dummy laser scan
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -math.pi/2
        scan_msg.angle_max = math.pi/2
        scan_msg.angle_increment = math.pi/180  # 1 degree
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        scan_msg.ranges = [5.0] * 181  # 181 points

        self.scan_pub.publish(scan_msg)

    def publish_navigation_status(self, status):
        """Publish navigation status"""
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)

    def publish_velocity_command(self, twist_msg):
        """Publish velocity command to robot"""
        # This would send the command to Isaac Sim robot
        pass

def main():
    """Main function to run Isaac Sim navigation integration"""
    print("Setting up Isaac Sim navigation environment...")

    # Create navigation interface
    nav_interface = IsaacSimNavigationInterface()

    # Setup Isaac Sim environment
    nav_interface.setup_isaac_sim_environment()

    # Run simulation
    print("Starting navigation simulation...")
    nav_interface.run_navigation_simulation()

    print("Navigation simulation completed.")

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid Navigation

### Design Guidelines

1. **Balance First**: Always prioritize dynamic balance over speed
2. **Gradual Transitions**: Smooth transitions between different gaits
3. **Robust Fallbacks**: Multiple recovery behaviors for different failure modes
4. **Modular Design**: Separate path planning, footstep planning, and balance control
5. **Real-time Performance**: Meet strict timing constraints for stability

### Safety Considerations

- **Emergency Stop**: Immediate halt on balance loss
- **Hardware Limits**: Respect joint limits and motor capabilities
- **Terrain Validation**: Verify walkability before execution
- **Sensor Validation**: Verify sensor data validity
- **State Monitoring**: Continuous monitoring of robot state

## Hands-On Exercise

### Exercise: Implementing Humanoid Navigation System

1. **Setup Navigation Environment**
   - Install Nav2 for ROS 2
   - Configure for humanoid-specific parameters
   - Set up Isaac Sim integration

2. **Implement Footstep Planning**
   - Create footstep planner with stability constraints
   - Integrate with path planning algorithms
   - Test with various terrain types

3. **Develop Balance Control**
   - Implement LQR or MPC balance controller
   - Test with perturbations and disturbances
   - Verify stability margins

4. **Integrate Navigation Stack**
   - Connect all components into complete system
   - Test with behavior trees
   - Validate real-time performance

5. **Deploy and Test**
   - Test in Isaac Sim environment
   - Evaluate navigation performance
   - Optimize for real-world deployment

## Summary

Nav2 provides a robust foundation for humanoid navigation, but requires significant adaptation for bipedal locomotion. The key challenges include footstep planning, balance control, and whole-body motion integration. By combining traditional path planning with humanoid-specific constraints and control algorithms, we can create navigation systems that enable safe and stable humanoid locomotion. Proper integration with simulation environments like Isaac Sim allows for thorough testing and validation before real-world deployment.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic Nav2 setup and simple path planning
- **Intermediate**: Dive deeper into footstep planning and balance control
- **Advanced**: Explore advanced MPC control and real-time optimization techniques