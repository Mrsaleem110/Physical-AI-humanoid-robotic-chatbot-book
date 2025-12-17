---
sidebar_position: 2
---

# ROS 2 Nodes, Topics, and Services

## Chapter Objectives

- Understand the fundamental ROS 2 communication patterns
- Create and run ROS 2 nodes
- Implement publisher-subscriber communication (topics)
- Implement client-server communication (services)

## ROS 2 Communication Patterns

ROS 2 provides three main communication patterns:

1. **Topics (Publish/Subscribe)**: Asynchronous, many-to-many communication
2. **Services (Request/Response)**: Synchronous, one-to-one communication
3. **Actions (Goal/Result/Feedback)**: Asynchronous, goal-oriented communication

## Nodes: The Basic Computational Units

A node is a process that performs computation. In ROS 2:

- Each node runs in its own process
- Nodes communicate with each other through topics, services, and actions
- Nodes are organized into packages

### Creating a Node

```python
# my_robot_package/my_robot_package/simple_node.py
import rclpy
from rclpy.node import Node

class SimpleNode(Node):
    def __init__(self):
        super().__init__('simple_node')
        self.get_logger().info('Simple Node Created')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Running a Node

```bash
# Make sure your workspace is sourced
source install/setup.bash

# Run the node
ros2 run my_robot_package simple_node
```

## Topics: Publish/Subscribe Communication

Topics enable asynchronous communication between nodes using a publish/subscribe model:

- Publishers send messages to topics
- Subscribers receive messages from topics
- Multiple publishers and subscribers can use the same topic
- Communication is asynchronous and non-blocking

### Publisher Example

```python
# my_robot_package/my_robot_package/talker.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()
```

### Subscriber Example

```python
# my_robot_package/my_robot_package/listener.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()
```

## Services: Request/Response Communication

Services provide synchronous request/response communication:

- Client sends a request to a service
- Server processes the request and sends a response
- Communication is synchronous and blocking

### Service Definition

```bash
# Create service definition file: srv/AddTwoInts.srv
int64 a
int64 b
---
int64 sum
```

### Service Server

```python
# my_robot_package/my_robot_package/add_two_ints_server.py
from my_robot_package.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsServer(Node):
    def __init__(self):
        super().__init__('add_two_ints_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: [{response.sum}]')
        return response

def main(args=None):
    rclpy.init(args=args)
    add_two_ints_server = AddTwoIntsServer()

    try:
        rclpy.spin(add_two_ints_server)
    except KeyboardInterrupt:
        pass
    finally:
        add_two_ints_server.destroy_node()
        rclpy.shutdown()
```

### Service Client

```python
# my_robot_package/my_robot_package/add_two_ints_client.py
from my_robot_package.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddTwoIntsClient(Node):
    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    add_two_ints_client = AddTwoIntsClient()
    response = add_two_ints_client.send_request(1, 2)
    add_two_ints_client.get_logger().info(f'Result of add_two_ints: {response.sum}')

    add_two_ints_client.destroy_node()
    rclpy.shutdown()
```

## ROS 2 Commands for Communication

### Topic Commands

```bash
# List all topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo /chatter std_msgs/msg/String

# Publish a message to a topic
ros2 topic pub /chatter std_msgs/msg/String "data: 'Hello World'"

# Get info about a topic
ros2 topic info /chatter
```

### Service Commands

```bash
# List all services
ros2 service list

# Call a service
ros2 service call /add_two_ints my_robot_package/srv/AddTwoInts "{a: 1, b: 2}"

# Get info about a service
ros2 service info /add_two_ints
```

## Best Practices for Humanoid Robots

### Topic Design

- Use appropriate message rates for real-time control
- Consider bandwidth limitations for wireless communication
- Use latching for static data that new subscribers need
- Implement QoS (Quality of Service) settings for reliability

### Service Design

- Use services for operations that should complete successfully
- Implement timeouts to prevent hanging
- Use actions instead of services for long-running operations
- Design services to be idempotent when possible

## Hands-On Exercise

1. Create a publisher that publishes joint positions for a humanoid robot
2. Create a subscriber that logs these positions
3. Create a service that calculates inverse kinematics for a target position
4. Test communication using ROS 2 command-line tools

## Summary

Understanding ROS 2 communication patterns is essential for building humanoid robots. Topics provide asynchronous data flow, while services provide synchronous request/response communication. In the next chapter, we'll explore Python integration with ROS 2.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on understanding the publish/subscribe model and basic node creation
- **Intermediate**: Dive deeper into QoS settings and message types
- **Advanced**: Explore custom message definitions and real-time considerations