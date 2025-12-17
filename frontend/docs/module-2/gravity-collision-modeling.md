---
sidebar_position: 2
---

# Gravity, Collision, and Environment Modeling

## Chapter Objectives

- Implement realistic gravity simulation for humanoid robots
- Model complex collision scenarios and responses
- Create diverse environments for robot testing
- Optimize collision detection for performance

## Gravity Simulation

### Understanding Gravity in Simulation

Gravity is a fundamental force that affects all objects in the simulation. For humanoid robots, gravity is crucial for:

- Balance and stability control
- Walking and locomotion algorithms
- Manipulation tasks
- Realistic physics interactions

### Configuring Gravity

In most simulation engines, gravity is configured globally:

```xml
<!-- Gazebo world file -->
<world name="gravity_world">
  <!-- Set gravity vector (x, y, z) in m/s^2 -->
  <gravity>0 0 -9.8</gravity>

  <!-- Physics engine configuration -->
  <physics name="default_physics" type="ode">
    <gravity>0 0 -9.8</gravity>
    <ode>
      <solver>
        <type>quick</type>
        <iters>10</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

### Gravity Considerations for Humanoid Robots

- **Walking Stability**: Gravity affects center of mass and balance
- **Foot Contact**: Proper contact with ground is essential for locomotion
- **Manipulation**: Gravity affects object handling and grasping
- **Energy Consumption**: Gravity impacts motor effort requirements

## Collision Detection in Depth

### Types of Collisions

1. **Self-Collision**: Robot parts colliding with each other
2. **Environment Collision**: Robot colliding with environment objects
3. **Object Collision**: Robot colliding with objects in the environment
4. **Multi-Body Collision**: Multiple robots or objects colliding

### Collision Detection Algorithms

#### Broad Phase (Culling)
```python
# Example broad phase collision detection
class BroadPhaseCollision:
    def __init__(self):
        self.spatial_grid = {}  # Grid for spatial partitioning
        self.bounding_boxes = []  # List of bounding boxes

    def find_potential_collisions(self):
        """Find pairs of objects that might be colliding"""
        potential_pairs = []

        # Use spatial partitioning to reduce comparisons
        for obj1 in self.bounding_boxes:
            nearby_objects = self.get_nearby_objects(obj1)
            for obj2 in nearby_objects:
                if self.bounding_boxes_collide(obj1, obj2):
                    potential_pairs.append((obj1, obj2))

        return potential_pairs
```

#### Narrow Phase (Precise Detection)
```python
# Example narrow phase collision detection
class NarrowPhaseCollision:
    def __init__(self):
        pass

    def detect_collision(self, shape1, shape2):
        """Detect precise collision between two shapes"""
        # For spheres
        if shape1.type == 'sphere' and shape2.type == 'sphere':
            return self.sphere_sphere_collision(shape1, shape2)
        # For boxes
        elif shape1.type == 'box' and shape2.type == 'box':
            return self.box_box_collision(shape1, shape2)
        # For mixed types
        else:
            return self.general_collision(shape1, shape2)

    def sphere_sphere_collision(self, sphere1, sphere2):
        """Detect collision between two spheres"""
        distance = self.calculate_distance(sphere1.center, sphere2.center)
        return distance < (sphere1.radius + sphere2.radius)
```

### Collision Response

When collisions are detected, the simulation must calculate the response:

```python
class CollisionResponse:
    def __init__(self):
        self.restitution = 0.5  # Bounciness (0 = no bounce, 1 = perfect bounce)
        self.friction = 0.8     # Friction coefficient

    def resolve_collision(self, obj1, obj2, contact_point, contact_normal):
        """Resolve collision between two objects"""
        # Calculate relative velocity at contact point
        rel_velocity = obj2.velocity - obj1.velocity

        # Calculate velocity along normal
        vel_along_normal = rel_velocity.dot(contact_normal)

        # Do not resolve if objects are moving apart
        if vel_along_normal > 0:
            return

        # Calculate restitution (bounciness)
        e = min(obj1.restitution, obj2.restitution)

        # Calculate impulse scalar
        j = -(1 + e) * vel_along_normal
        j /= 1 / obj1.mass + 1 / obj2.mass

        # Apply impulse
        impulse = j * contact_normal
        obj1.velocity -= impulse / obj1.mass
        obj2.velocity += impulse / obj2.mass
```

## Environment Modeling for Humanoid Robots

### Indoor Environments

For humanoid robots, indoor environments often include:

- **Flat Surfaces**: Floors, platforms, stages
- **Obstacles**: Furniture, walls, doors
- **Stairs**: Multi-level navigation challenges
- **Narrow Passages**: Doorways, corridors
- **Interactive Objects**: Tables, chairs, switches

### Creating Complex Environments

```xml
<!-- Gazebo world with complex indoor environment -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="complex_indoor">
    <!-- Gravity -->
    <gravity>0 0 -9.8</gravity>

    <!-- Physics -->
    <physics name="default_physics" type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room structure -->
    <model name="room_walls">
      <!-- Wall 1 -->
      <link name="wall1">
        <pose>0 -5 2.5 0 0 0</pose>
        <collision name="wall1_collision">
          <geometry>
            <box>
              <size>20 0.2 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall1_visual">
          <geometry>
            <box>
              <size>20 0.2 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Other walls, floor, ceiling would be defined similarly -->
    </model>

    <!-- Furniture -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_top">
        <collision name="table_collision">
          <geometry>
            <box>
              <size>1.5 0.8 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="table_visual">
          <geometry>
            <box>
              <size>1.5 0.8 0.02</size>
            </box>
          </geometry>
        </visual>
      </link>

      <!-- Table legs -->
      <model name="leg1">
        <pose>0.6 0.3 -0.39 0 0 0</pose>
        <!-- Leg geometry -->
      </model>
      <!-- More legs... -->
    </model>

    <!-- Stairs -->
    <model name="stairs">
      <pose>-3 -2 0 0 0 0</pose>
      <!-- Define multiple steps -->
      <link name="step1">
        <pose>0 0 0.15 0 0 0</pose>
        <collision name="step1_collision">
          <geometry>
            <box>
              <size>2 1 0.3</size>
            </box>
          </geometry>
        </collision>
        <visual name="step1_visual">
          <geometry>
            <box>
              <size>2 1 0.3</size>
            </box>
          </geometry>
        </visual>
      </link>
      <!-- More steps... -->
    </model>
  </world>
</sdf>
```

### Outdoor Environments

Outdoor environments for humanoid robots might include:

- **Terrain**: Hills, slopes, uneven ground
- **Natural Obstacles**: Rocks, trees, bushes
- **Weather Effects**: Rain, wind (simulation of effects)
- **Dynamic Elements**: Moving vehicles, other agents

## Collision Optimization for Performance

### Level of Detail (LOD)

```python
class CollisionLOD:
    def __init__(self):
        self.detail_levels = {
            'high': {'sphere_count': 100, 'box_subdivisions': 10},
            'medium': {'sphere_count': 50, 'box_subdivisions': 5},
            'low': {'sphere_count': 20, 'box_subdivisions': 2}
        }

    def get_collision_model(self, distance, complexity_level):
        """Get appropriate collision model based on distance and complexity"""
        if distance > 10:  # Far away
            lod_level = 'low'
        elif distance > 5:  # Medium distance
            lod_level = 'medium'
        else:  # Close up
            lod_level = 'high'

        return self.create_collision_model(lod_level, complexity_level)

    def create_collision_model(self, lod_level, complexity_level):
        """Create collision model based on LOD and complexity"""
        # Implementation would create simplified collision geometry
        pass
```

### Spatial Partitioning

```python
class SpatialPartitioning:
    def __init__(self, world_size, cell_size):
        self.world_size = world_size
        self.cell_size = cell_size
        self.grid = self.create_grid()

    def create_grid(self):
        """Create spatial partitioning grid"""
        grid_size = int(self.world_size / self.cell_size)
        return [[[] for _ in range(grid_size)] for _ in range(grid_size)]

    def add_object(self, obj, position):
        """Add object to appropriate grid cell"""
        grid_x = int(position[0] / self.cell_size)
        grid_y = int(position[1] / self.cell_size)

        if 0 <= grid_x < len(self.grid) and 0 <= grid_y < len(self.grid[0]):
            self.grid[grid_x][grid_y].append(obj)

    def get_nearby_objects(self, position, radius=1.0):
        """Get objects in nearby grid cells"""
        nearby_objects = []

        # Calculate grid range
        grid_x = int(position[0] / self.cell_size)
        grid_y = int(position[1] / self.cell_size)

        range_cells = int(radius / self.cell_size) + 1

        for dx in range(-range_cells, range_cells + 1):
            for dy in range(-range_cells, range_cells + 1):
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < len(self.grid) and 0 <= ny < len(self.grid[0]):
                    nearby_objects.extend(self.grid[nx][ny])

        return nearby_objects
```

## Advanced Collision Scenarios for Humanoid Robots

### Balance and Stability

Humanoid robots must maintain balance under various conditions:

```python
class BalanceController:
    def __init__(self):
        self.com_height = 0.8  # Center of mass height
        self.support_polygon = []  # Area where CoM must stay

    def check_stability(self, robot_pose, foot_positions):
        """Check if robot is stable based on CoM position"""
        # Calculate center of mass projection on ground
        com_projection = self.calculate_com_projection(robot_pose)

        # Calculate support polygon from foot positions
        support_polygon = self.calculate_support_polygon(foot_positions)

        # Check if CoM is within support polygon
        is_stable = self.point_in_polygon(com_projection, support_polygon)

        return is_stable, support_polygon

    def calculate_support_polygon(self, foot_positions):
        """Calculate support polygon from foot contact points"""
        # For bipedal robot, this is typically a polygon connecting foot points
        if len(foot_positions) >= 2:
            # Create convex hull of foot positions
            return self.convex_hull(foot_positions)
        else:
            # Single foot support
            return [foot_positions[0]]
```

### Manipulation and Grasping

Collision detection is crucial for manipulation tasks:

```python
class ManipulationController:
    def __init__(self):
        self.gripper_open = True
        self.object_grasped = None

    def attempt_grasp(self, gripper_pose, object_pose):
        """Attempt to grasp an object"""
        # Check if gripper is close enough to object
        distance = self.calculate_distance(gripper_pose, object_pose)

        if distance < 0.05 and self.gripper_open:  # Within grasp distance
            # Check for proper orientation
            if self.check_grasp_orientation(gripper_pose, object_pose):
                # Check for collision-free grasp
                if not self.would_collide_during_grasp(gripper_pose, object_pose):
                    self.object_grasped = object_pose
                    self.gripper_open = False
                    return True, "Grasp successful"

        return False, "Grasp failed"

    def would_collide_during_grasp(self, gripper_pose, object_pose):
        """Check if grasp would cause collisions"""
        # Simulate the grasp motion and check for collisions
        # with environment and robot self-collision
        pass
```

## Environment Complexity Management

### Adaptive Complexity

```python
class EnvironmentComplexityManager:
    def __init__(self):
        self.current_complexity = 'medium'
        self.performance_threshold = 0.8  # Target performance ratio

    def adjust_environment_complexity(self, current_performance):
        """Adjust environment complexity based on performance"""
        if current_performance < self.performance_threshold * 0.8:
            # Performance too low, reduce complexity
            self.reduce_complexity()
        elif current_performance > self.performance_threshold * 1.2:
            # Performance high, can increase complexity
            self.increase_complexity()

    def reduce_complexity(self):
        """Reduce environment complexity"""
        if self.current_complexity == 'high':
            self.current_complexity = 'medium'
            self.simplify_collision_meshes()
        elif self.current_complexity == 'medium':
            self.current_complexity = 'low'
            self.remove_decoration_objects()

    def increase_complexity(self):
        """Increase environment complexity"""
        if self.current_complexity == 'low':
            self.current_complexity = 'medium'
            self.add_decoration_objects()
        elif self.current_complexity == 'medium':
            self.current_complexity = 'high'
            self.increase_collision_detail()
```

## Best Practices for Collision Modeling

### Robot Self-Collision

- Define self-collision pairs that should be ignored (adjacent links)
- Use simplified collision geometry for self-collision detection
- Implement efficient self-collision culling

### Environment Interaction

- Use appropriate restitution values for different materials
- Implement friction anisotropy for directional friction effects
- Consider contact surface properties (sticky, slippery, etc.)

### Performance Considerations

- Use compound collision shapes to reduce primitive count
- Implement temporal coherence to optimize repeated collision queries
- Balance visual fidelity with collision performance

## Hands-On Exercise

1. Create a complex indoor environment with furniture and obstacles
2. Implement collision detection between a humanoid robot and environment
3. Test the robot's balance under different gravity conditions
4. Optimize collision detection for performance

## Summary

Gravity and collision modeling are fundamental to realistic humanoid robot simulation. Proper implementation of these physics concepts enables safe and effective testing of robot behaviors. Understanding how to optimize collision detection for performance while maintaining accuracy is crucial for complex simulation scenarios. In the next chapter, we'll explore Unity-based HRI visualization.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic collision detection and simple environments
- **Intermediate**: Dive deeper into optimization techniques and complex scenarios
- **Advanced**: Explore advanced physics concepts and real-time performance optimization