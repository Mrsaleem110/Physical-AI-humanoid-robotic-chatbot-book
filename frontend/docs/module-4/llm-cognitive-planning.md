---
sidebar_position: 2
---

# LLM-based Cognitive Planning (NL → ROS 2 Actions)

## Chapter Objectives

- Understand Large Language Model integration with robotics systems
- Implement natural language to ROS 2 action mapping
- Create cognitive planning pipelines for humanoid robots
- Design context-aware reasoning systems
- Optimize LLM responses for real-time robotic applications

## Introduction to LLM-Robotics Integration

Large Language Models (LLMs) have revolutionized how we can bridge natural language understanding with robotic action execution. For humanoid robotics, LLMs serve as cognitive engines that can:

- **Interpret Natural Language**: Understand complex, multi-step commands
- **Plan Actions**: Generate sequences of ROS 2 actions from high-level goals
- **Reason Contextually**: Adapt behavior based on environmental context
- **Handle Ambiguity**: Resolve unclear commands through clarification
- **Learn from Interaction**: Improve over time through user feedback

### Why LLMs for Robotics?

1. **Natural Interaction**: Enable human-like communication with robots
2. **Complex Reasoning**: Handle complex, multi-step tasks
3. **Adaptability**: Adapt to new situations and commands
4. **Knowledge Integration**: Leverage pre-trained world knowledge
5. **Scalability**: Handle diverse command types without explicit programming

### LLM Integration Challenges

- **Latency**: Ensuring real-time response for robotic actions
- **Reliability**: Handling uncertain LLM outputs in safety-critical applications
- **Context Management**: Maintaining conversation and task context
- **Action Grounding**: Mapping abstract concepts to concrete robot actions
- **Safety**: Ensuring safe execution of LLM-generated commands

## LLM Architecture for Robotics

### Cognitive Planning Framework

```python
# python/llm_robotics_framework.py
import openai
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time
from enum import Enum
import logging

class ActionStatus(Enum):
    """Status of robot actions"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """Types of tasks that can be generated"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INTERACTION = "interaction"
    PERCEPTION = "perception"
    COMPOSITE = "composite"

@dataclass
class RobotAction:
    """Represents a single robot action"""
    action_type: str  # e.g., "move_to", "pick_up", "greet"
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    dependencies: List[str] = None

@dataclass
class CognitiveTask:
    """Represents a high-level cognitive task"""
    id: str
    description: str
    task_type: TaskType
    actions: List[RobotAction]
    context: Dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING
    created_at: float = time.time()

class LLMRobotInterface:
    """Interface between LLM and robotic systems"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.task_history = []

        # Context management
        self.conversation_context = []
        self.robot_state = {}
        self.environment_map = {}
        self.object_database = {}

        # ROS 2 integration
        self.ros_interface = None

        # Statistics
        self.stats = {
            'total_requests': 0,
            'average_response_time': 0.0,
            'success_rate': 0.0,
            'context_size': 0
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"LLM Robot Interface initialized with model: {model}")

    async def process_natural_language(self, command: str, context: Dict[str, Any] = None) -> CognitiveTask:
        """Process natural language command and generate cognitive task"""
        start_time = time.time()

        # Build context for LLM
        llm_context = self._build_llm_context(command, context)

        # Generate action plan using LLM
        action_plan = await self._generate_action_plan(llm_context)

        # Parse and validate action plan
        cognitive_task = self._parse_action_plan(action_plan, command)

        # Update statistics
        response_time = time.time() - start_time
        self.stats['total_requests'] += 1
        self.stats['average_response_time'] = (
            (self.stats['average_response_time'] * (self.stats['total_requests'] - 1) + response_time) /
            self.stats['total_requests']
        )

        # Add to conversation context
        self.conversation_context.append({
            'role': 'user',
            'content': command,
            'timestamp': start_time
        })
        self.conversation_context.append({
            'role': 'assistant',
            'content': str(action_plan),
            'timestamp': time.time()
        })

        # Limit context size to prevent memory issues
        if len(self.conversation_context) > 20:
            self.conversation_context = self.conversation_context[-20:]

        self.stats['context_size'] = len(self.conversation_context)

        return cognitive_task

    def _build_llm_context(self, command: str, additional_context: Dict[str, Any] = None) -> str:
        """Build context for LLM including robot state and environment"""
        context_parts = []

        # Add robot capabilities
        capabilities = [
            "navigation: move_to(x, y, theta)",
            "manipulation: pick_up(object), place_at(location)",
            "perception: detect_object(type), locate_object(name)",
            "interaction: speak(text), gesture(type)",
            "safety: stop_immediately(), emergency_halt()"
        ]
        context_parts.append(f"Robot capabilities: {', '.join(capabilities)}")

        # Add current robot state
        if self.robot_state:
            context_parts.append(f"Current robot state: {json.dumps(self.robot_state)}")

        # Add environment information
        if self.environment_map:
            context_parts.append(f"Environment map: {json.dumps(self.environment_map)}")

        # Add known objects
        if self.object_database:
            context_parts.append(f"Known objects: {json.dumps(list(self.object_database.keys()))}")

        # Add additional context if provided
        if additional_context:
            context_parts.append(f"Additional context: {json.dumps(additional_context)}")

        # Add the user command
        context_parts.append(f"User command: {command}")

        # Add action format requirements
        context_parts.append(
            "Response format: Provide a JSON plan with 'actions' array. "
            "Each action should have 'type' and 'parameters'. "
            "Example: {\"actions\": [{\"type\": \"move_to\", \"parameters\": {\"x\": 1.0, \"y\": 2.0}}]}"
        )

        return "\n".join(context_parts)

    async def _generate_action_plan(self, context: str) -> Dict[str, Any]:
        """Generate action plan using LLM"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cognitive planning assistant for a humanoid robot. "
                            "Your job is to interpret natural language commands and generate "
                            "executable action plans. Always respond with valid JSON containing "
                            "an 'actions' array. Each action should have 'type' and 'parameters'."
                        )
                    },
                    {
                        "role": "user",
                        "content": context
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=1000,
                timeout=30
            )

            # Extract and parse the response
            content = response.choices[0].message.content.strip()

            # Clean up the response (remove any markdown formatting)
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```

            action_plan = json.loads(content)
            return action_plan

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return a default action plan
            return {"actions": [{"type": "speak", "parameters": {"text": "I didn't understand that command."}}]}

        except Exception as e:
            self.logger.error(f"Error generating action plan: {e}")
            return {"actions": [{"type": "speak", "parameters": {"text": "I'm having trouble processing that command."}}]}

    def _parse_action_plan(self, action_plan: Dict[str, Any], original_command: str) -> CognitiveTask:
        """Parse LLM-generated action plan into cognitive task"""
        actions = []

        if "actions" in action_plan:
            for action_data in action_plan["actions"]:
                action_type = action_data.get("type", "unknown")
                parameters = action_data.get("parameters", {})

                # Validate and normalize action
                normalized_action = self._validate_action(action_type, parameters)
                if normalized_action:
                    actions.append(normalized_action)

        # Determine task type based on actions
        task_type = self._determine_task_type(actions)

        # Create cognitive task
        task_id = f"task_{int(time.time())}_{len(self.task_history)}"
        cognitive_task = CognitiveTask(
            id=task_id,
            description=original_command,
            task_type=task_type,
            actions=actions,
            context={"original_command": original_command, "plan": action_plan}
        )

        self.task_history.append(cognitive_task)
        return cognitive_task

    def _validate_action(self, action_type: str, parameters: Dict[str, Any]) -> Optional[RobotAction]:
        """Validate and normalize robot action"""
        # Define valid action types and their required parameters
        valid_actions = {
            "move_to": {"x", "y", "theta"},
            "pick_up": {"object"},
            "place_at": {"location"},
            "detect_object": {"type"},
            "locate_object": {"name"},
            "speak": {"text"},
            "gesture": {"type"},
            "stop_immediately": set(),
            "emergency_halt": set(),
            "follow": {"target"},
            "search": {"object"},
            "bring_to": {"object", "destination"}
        }

        if action_type not in valid_actions:
            self.logger.warning(f"Invalid action type: {action_type}")
            return None

        required_params = valid_actions[action_type]

        # Check if all required parameters are present
        missing_params = required_params - set(parameters.keys())
        if missing_params:
            self.logger.warning(f"Missing required parameters for {action_type}: {missing_params}")
            return None

        # Normalize parameters
        normalized_params = parameters.copy()

        # Convert string numbers to floats where appropriate
        for param in ["x", "y", "theta"]:
            if param in normalized_params and isinstance(normalized_params[param], str):
                try:
                    normalized_params[param] = float(normalized_params[param])
                except ValueError:
                    self.logger.warning(f"Invalid numeric value for {param}: {normalized_params[param]}")
                    return None

        return RobotAction(
            action_type=action_type,
            parameters=normalized_params
        )

    def _determine_task_type(self, actions: List[RobotAction]) -> TaskType:
        """Determine task type based on actions"""
        action_types = [action.action_type for action in actions]

        if any(action_type in ["move_to", "follow"] for action_type in action_types):
            return TaskType.NAVIGATION
        elif any(action_type in ["pick_up", "place_at", "bring_to"] for action_type in action_types):
            return TaskType.MANIPULATION
        elif any(action_type in ["speak", "gesture"] for action_type in action_types):
            return TaskType.INTERACTION
        elif any(action_type in ["detect_object", "locate_object", "search"] for action_type in action_types):
            return TaskType.PERCEPTION
        else:
            return TaskType.COMPOSITE

    def update_robot_state(self, state: Dict[str, Any]):
        """Update robot state for context"""
        self.robot_state.update(state)

    def update_environment_map(self, env_map: Dict[str, Any]):
        """Update environment map for context"""
        self.environment_map.update(env_map)

    def add_known_object(self, obj_name: str, obj_info: Dict[str, Any]):
        """Add known object to database"""
        self.object_database[obj_name] = obj_info

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()

class TaskExecutor:
    """Execute cognitive tasks on the robot"""

    def __init__(self, llm_interface: LLMRobotInterface):
        self.llm_interface = llm_interface
        self.active_task = None
        self.executor_thread = None
        self.is_running = False

    def start_execution(self):
        """Start task execution loop"""
        self.is_running = True
        self.executor_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.executor_thread.start()

    def stop_execution(self):
        """Stop task execution"""
        self.is_running = False
        if self.executor_thread:
            self.executor_thread.join()

    def _execution_loop(self):
        """Main execution loop"""
        while self.is_running:
            try:
                # Check for new tasks
                if not self.llm_interface.task_queue.empty():
                    task = self.llm_interface.task_queue.get_nowait()
                    self.execute_task(task)

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except asyncio.QueueEmpty:
                time.sleep(0.1)
            except Exception as e:
                self.llm_interface.logger.error(f"Error in execution loop: {e}")
                time.sleep(0.1)

    def execute_task(self, task: CognitiveTask):
        """Execute a cognitive task"""
        self.active_task = task
        task.status = ActionStatus.EXECUTING

        self.llm_interface.logger.info(f"Executing task {task.id}: {task.description}")

        try:
            for action in task.actions:
                if not self._execute_action(action):
                    task.status = ActionStatus.FAILED
                    break

            if task.status != ActionStatus.FAILED:
                task.status = ActionStatus.SUCCESS

        except Exception as e:
            self.llm_interface.logger.error(f"Error executing task {task.id}: {e}")
            task.status = ActionStatus.FAILED

        task.completed_at = time.time()
        self.active_task = None

    def _execute_action(self, action: RobotAction) -> bool:
        """Execute a single robot action"""
        self.llm_interface.logger.info(f"Executing action: {action.action_type} with params: {action.parameters}")

        # In a real implementation, this would interface with ROS 2
        # For now, we'll simulate the execution
        success = self._simulate_action_execution(action)

        # Add delay based on action type
        action_time = self._estimate_action_time(action)
        time.sleep(min(action_time, 5.0))  # Cap at 5 seconds

        return success

    def _simulate_action_execution(self, action: RobotAction) -> bool:
        """Simulate action execution"""
        # This would interface with actual robot hardware in a real system
        # For simulation, we'll return success for most actions
        if action.action_type == "move_to":
            # Simulate navigation
            x = action.parameters.get('x', 0)
            y = action.parameters.get('y', 0)
            self.llm_interface.logger.info(f"Simulating move to ({x}, {y})")
        elif action.action_type == "pick_up":
            obj = action.parameters.get('object', 'unknown')
            self.llm_interface.logger.info(f"Simulating pick up of {obj}")
        elif action.action_type == "speak":
            text = action.parameters.get('text', '')
            self.llm_interface.logger.info(f"Simulating speech: {text}")
        elif action.action_type == "detect_object":
            obj_type = action.parameters.get('type', 'unknown')
            self.llm_interface.logger.info(f"Simulating object detection: {obj_type}")

        return True  # Assume success for simulation

    def _estimate_action_time(self, action: RobotAction) -> float:
        """Estimate time required for action execution"""
        time_estimates = {
            "move_to": 3.0,
            "pick_up": 5.0,
            "place_at": 4.0,
            "detect_object": 2.0,
            "locate_object": 3.0,
            "speak": 1.0,
            "gesture": 1.5,
            "stop_immediately": 0.1,
            "emergency_halt": 0.1,
            "follow": 10.0,
            "search": 8.0,
            "bring_to": 15.0
        }
        return time_estimates.get(action.action_type, 2.0)

def main():
    """Main function to demonstrate LLM-robotics integration"""
    print("LLM Robotics Interface Demo")

    # Note: In a real implementation, you would provide your OpenAI API key
    # For this example, we'll show the structure
    try:
        # Initialize LLM interface (with a placeholder API key)
        llm_interface = LLMRobotInterface(
            api_key="YOUR_OPENAI_API_KEY_HERE",  # Replace with actual API key
            model="gpt-4-turbo"
        )

        # Example commands to test
        test_commands = [
            "Move to the kitchen and bring me a cup",
            "Find the red ball and pick it up",
            "Go to the living room and greet the person there",
            "Locate the book on the table and move it to the shelf"
        ]

        print("\nTesting natural language processing:")
        for command in test_commands:
            print(f"\nProcessing: '{command}'")

            # Process the command
            task = asyncio.run(llm_interface.process_natural_language(command))

            print(f"Generated task: {task.task_type.value}")
            print(f"Actions: {[action.action_type for action in task.actions]}")

        print(f"\nStatistics: {llm_interface.get_stats()}")

    except Exception as e:
        print(f"Error initializing LLM interface: {e}")
        print("Make sure you have OpenAI API key and proper setup")

if __name__ == "__main__":
    main()
```

## Natural Language Understanding for Robotics

### Context-Aware Processing

```python
# python/context_aware_processing.py
import re
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

class ContextType(Enum):
    """Types of context in robotic interactions"""
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    OBJECT = "object"
    AGENT = "agent"
    TASK = "task"
    ENVIRONMENTAL = "environmental"

@dataclass
class ContextItem:
    """Represents a context item with type, value, and temporal information"""
    context_type: ContextType
    value: Any
    timestamp: float
    confidence: float = 1.0
    source: str = "unknown"

class ContextManager:
    """Manages context for LLM-based robotic planning"""

    def __init__(self, max_context_items: int = 100):
        self.max_context_items = max_context_items
        self.context_items: List[ContextItem] = []
        self.context_by_type: Dict[ContextType, List[ContextItem]] = {
            ctx_type: [] for ctx_type in ContextType
        }

    def add_context(self, context_type: ContextType, value: Any, confidence: float = 1.0, source: str = "unknown"):
        """Add a context item"""
        context_item = ContextItem(
            context_type=context_type,
            value=value,
            timestamp=time.time(),
            confidence=confidence,
            source=source
        )

        self.context_items.append(context_item)
        self.context_by_type[context_type].append(context_item)

        # Maintain size limits
        if len(self.context_items) > self.max_context_items:
            self._prune_context()

    def get_context_by_type(self, context_type: ContextType) -> List[ContextItem]:
        """Get context items of a specific type"""
        return self.context_by_type[context_type]

    def get_recent_context(self, time_window: float = 300.0) -> List[ContextItem]:  # 5 minutes
        """Get context items from recent time window"""
        current_time = time.time()
        recent_items = [
            item for item in self.context_items
            if (current_time - item.timestamp) <= time_window
        ]
        return recent_items

    def get_spatial_context(self) -> Dict[str, Any]:
        """Get spatial context (locations, coordinates, etc.)"""
        spatial_items = self.get_context_by_type(ContextType.SPATIAL)
        spatial_dict = {}

        for item in spatial_items:
            if isinstance(item.value, dict):
                spatial_dict.update(item.value)

        return spatial_dict

    def get_object_context(self) -> Dict[str, Any]:
        """Get object context (known objects, their properties, etc.)"""
        object_items = self.get_context_by_type(ContextType.OBJECT)
        object_dict = {}

        for item in object_items:
            if isinstance(item.value, dict):
                object_dict.update(item.value)

        return object_dict

    def _prune_context(self):
        """Prune context to maintain size limits"""
        # Remove oldest items first
        self.context_items.sort(key=lambda x: x.timestamp)
        excess = len(self.context_items) - self.max_context_items
        if excess > 0:
            removed_items = self.context_items[:excess]
            self.context_items = self.context_items[excess:]

        # Update context_by_type
        self.context_by_type = {ctx_type: [] for ctx_type in ContextType}
        for item in self.context_items:
            self.context_by_type[item.context_type].append(item)

    def clear_context(self):
        """Clear all context"""
        self.context_items = []
        self.context_by_type = {ctx_type: [] for ctx_type in ContextType}

class NaturalLanguageProcessor:
    """Process natural language with context awareness"""

    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

        # Define spatial reference patterns
        self.spatial_patterns = {
            'relative_directions': [
                (r'to the (left|right|front|back|behind|in front of)', r'@\1'),
                (r'go (north|south|east|west)', r'@\1'),
                (r'move (forward|backward)', r'@\1'),
            ],
            'locations': [
                (r'to the (\w+ room)', r'@\1'),
                (r'in the (\w+)', r'@\1'),
                (r'at the (\w+)', r'@\1'),
            ],
            'objects': [
                (r'the (\w+ \w+)', r'@\1'),  # "the red ball"
                (r'a (\w+)', r'@a_\1'),
                (r'an (\w+)', r'@an_\1'),
            ]
        }

        # Define temporal patterns
        self.temporal_patterns = [
            r'now',
            r'immediately',
            r'right now',
            r'as soon as possible',
            r'after',
            r'before',
            r'when',
            r'until'
        ]

    def process_command(self, command: str) -> Tuple[str, Dict[str, Any]]:
        """Process natural language command with context resolution"""
        # Extract spatial references
        resolved_command, spatial_refs = self._resolve_spatial_references(command)

        # Extract temporal references
        temporal_refs = self._extract_temporal_references(command)

        # Extract object references
        object_refs = self._extract_object_references(command)

        # Combine all references
        all_refs = {
            'spatial': spatial_refs,
            'temporal': temporal_refs,
            'objects': object_refs
        }

        return resolved_command, all_refs

    def _resolve_spatial_references(self, command: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Resolve spatial references in command using context"""
        resolved_command = command.lower()
        spatial_references = []

        # Apply spatial patterns
        for pattern_type, patterns in self.spatial_patterns.items():
            for pattern, replacement in patterns:
                matches = re.finditer(pattern, resolved_command, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(0)
                    entity = match.group(1) if len(match.groups()) > 0 else matched_text

                    # Look up in context if needed
                    if pattern_type == 'locations':
                        location_info = self._lookup_location(entity)
                        if location_info:
                            spatial_references.append({
                                'type': pattern_type,
                                'entity': entity,
                                'resolved': location_info,
                                'original': matched_text
                            })
                    elif pattern_type == 'objects':
                        object_info = self._lookup_object(entity)
                        if object_info:
                            spatial_references.append({
                                'type': pattern_type,
                                'entity': entity,
                                'resolved': object_info,
                                'original': matched_text
                            })

        return resolved_command, spatial_references

    def _extract_temporal_references(self, command: str) -> List[str]:
        """Extract temporal references from command"""
        temporal_refs = []
        for pattern in self.temporal_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                temporal_refs.append(pattern)
        return temporal_refs

    def _extract_object_references(self, command: str) -> List[Dict[str, str]]:
        """Extract object references from command"""
        object_refs = []

        # Look for object patterns in the command
        object_patterns = [
            r'the (\w+ \w+)',  # "the red ball"
            r'the (\w+)',      # "the ball"
            r'a (\w+)',        # "a ball"
            r'an (\w+)',       # "an object"
        ]

        for pattern in object_patterns:
            matches = re.finditer(pattern, command, re.IGNORECASE)
            for match in matches:
                object_name = match.group(1)
                object_refs.append({
                    'raw': match.group(0),
                    'name': object_name,
                    'type': self._infer_object_type(object_name)
                })

        return object_refs

    def _lookup_location(self, location_name: str) -> Optional[Dict[str, Any]]:
        """Look up location information in context"""
        spatial_context = self.context_manager.get_spatial_context()
        return spatial_context.get(location_name.lower())

    def _lookup_object(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Look up object information in context"""
        object_context = self.context_manager.get_object_context()
        return object_context.get(object_name.lower())

    def _infer_object_type(self, object_name: str) -> str:
        """Infer object type from name"""
        # Simple heuristic-based inference
        object_name_lower = object_name.lower()

        if any(word in object_name_lower for word in ['ball', 'cup', 'bottle', 'box']):
            return 'graspable'
        elif any(word in object_name_lower for word in ['person', 'human', 'man', 'woman']):
            return 'agent'
        elif any(word in object_name_lower for word in ['table', 'chair', 'shelf', 'cabinet']):
            return 'furniture'
        elif any(word in object_name_lower for word in ['door', 'window', 'hallway', 'kitchen']):
            return 'location'
        else:
            return 'unknown'

class ContextualCommandResolver:
    """Resolve ambiguous commands using context"""

    def __init__(self, context_manager: ContextManager, nlp_processor: NaturalLanguageProcessor):
        self.context_manager = context_manager
        self.nlp_processor = nlp_processor

    def resolve_command(self, command: str, user_intent: str = None) -> str:
        """Resolve ambiguous command using context"""
        # Process the command to extract references
        resolved_command, references = self.nlp_processor.process_command(command)

        # Resolve spatial ambiguities
        resolved_command = self._resolve_spatial_ambiguities(resolved_command, references)

        # Resolve object ambiguities
        resolved_command = self._resolve_object_ambiguities(resolved_command, references)

        # Add context to the resolved command
        contextual_command = self._add_contextual_info(resolved_command, user_intent)

        return contextual_command

    def _resolve_spatial_ambiguities(self, command: str, references: Dict[str, Any]) -> str:
        """Resolve spatial ambiguities using context"""
        resolved_cmd = command

        # Replace ambiguous spatial references with specific coordinates or locations
        for ref in references.get('spatial', []):
            if ref['type'] == 'locations' and 'resolved' in ref:
                location_info = ref['resolved']
                if 'coordinates' in location_info:
                    coords = location_info['coordinates']
                    replacement = f"move_to(x={coords[0]}, y={coords[1]}, theta={coords[2]})"
                    resolved_cmd = resolved_cmd.replace(ref['original'], replacement)

        return resolved_cmd

    def _resolve_object_ambiguities(self, command: str, references: Dict[str, Any]) -> str:
        """Resolve object ambiguities using context"""
        resolved_cmd = command

        # Replace ambiguous object references with specific object information
        for ref in references.get('objects', []):
            if 'resolved' in ref:
                obj_info = ref['resolved']
                if 'id' in obj_info:
                    replacement = f"object_id_{obj_info['id']}"
                    resolved_cmd = resolved_cmd.replace(ref['name'], replacement)

        return resolved_cmd

    def _add_contextual_info(self, command: str, user_intent: str = None) -> str:
        """Add contextual information to command"""
        # Add recent context to help LLM understand
        recent_context = self.context_manager.get_recent_context(time_window=600)  # 10 minutes

        context_summary = {
            'recent_locations': [item.value for item in recent_context
                               if item.context_type == ContextType.SPATIAL],
            'recent_objects': [item.value for item in recent_context
                             if item.context_type == ContextType.OBJECT],
            'recent_actions': [item.value for item in recent_context
                             if item.context_type == ContextType.TASK]
        }

        # Add context to command
        if user_intent:
            contextual_command = f"User intent: {user_intent}\nCommand: {command}\nContext: {json.dumps(context_summary)}"
        else:
            contextual_command = f"Command: {command}\nContext: {json.dumps(context_summary)}"

        return contextual_command

def demonstrate_contextual_processing():
    """Demonstrate contextual command processing"""
    print("Demonstrating Context-Aware Natural Language Processing")

    # Initialize context manager
    context_manager = ContextManager()

    # Add some context
    context_manager.add_context(
        ContextType.SPATIAL,
        {"kitchen": {"coordinates": [2.0, 3.0, 0.0], "description": "cooking area"}},
        confidence=0.9
    )

    context_manager.add_context(
        ContextType.OBJECT,
        {"red_ball": {"id": "obj_001", "type": "graspable", "location": [1.5, 2.0, 0.0]}},
        confidence=0.8
    )

    # Initialize NLP processor
    nlp_processor = NaturalLanguageProcessor(context_manager)

    # Initialize command resolver
    resolver = ContextualCommandResolver(context_manager, nlp_processor)

    # Test commands
    test_commands = [
        "Go to the kitchen",
        "Pick up the red ball",
        "Move to the table",
        "Find the book"
    ]

    print("\nProcessing commands with context resolution:")
    for cmd in test_commands:
        print(f"\nOriginal: '{cmd}'")
        resolved = resolver.resolve_command(cmd)
        print(f"Resolved: '{resolved[:100]}...'" if len(resolved) > 100 else f"Resolved: '{resolved}'")

if __name__ == "__main__":
    demonstrate_contextual_processing()
```

## Cognitive Planning Algorithms

### Hierarchical Task Planning

```python
# python/hierarchical_planning.py
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import time

class TaskStatus(Enum):
    """Status of hierarchical tasks"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 5
    HIGH = 10
    CRITICAL = 15

@dataclass
class PrimitiveAction:
    """Lowest level action that can be executed directly"""
    name: str
    parameters: Dict[str, Any]
    duration: float  # Estimated execution time in seconds
    preconditions: List[str]  # Conditions that must be true before execution
    effects: List[str]  # Effects of the action on the world state

@dataclass
class CompoundTask:
    """A task composed of subtasks"""
    name: str
    description: str
    subtasks: List['TaskNode']
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: float = time.time()
    estimated_duration: float = 0.0

@dataclass
class TaskNode:
    """Node in the hierarchical task tree"""
    task_id: str
    task_type: str  # "primitive" or "compound"
    content: Any  # Either PrimitiveAction or CompoundTask
    parent: Optional['TaskNode'] = None
    children: List['TaskNode'] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = None  # Task IDs this task depends on
    priority: TaskPriority = TaskPriority.MEDIUM

class HierarchicalPlanner:
    """Implements hierarchical task planning for robotics"""

    def __init__(self):
        self.task_tree: Optional[TaskNode] = None
        self.task_registry: Dict[str, TaskNode] = {}
        self.world_state: Dict[str, Any] = {}
        self.executor = TaskExecutor()

    def create_plan(self, goal: str, context: Dict[str, Any]) -> TaskNode:
        """Create a hierarchical plan for the given goal"""
        # This would use an LLM to decompose the high-level goal
        # into a hierarchical task structure
        plan = self._decompose_goal(goal, context)
        self.task_tree = plan
        return plan

    def _decompose_goal(self, goal: str, context: Dict[str, Any]) -> TaskNode:
        """Decompose high-level goal into hierarchical tasks"""
        # In a real implementation, this would call an LLM to generate the decomposition
        # For this example, we'll create a simple decomposition

        # Example: "Bring me a cup of water from the kitchen"
        root_task = CompoundTask(
            name="bring_water",
            description=goal,
            subtasks=[]
        )

        root_node = TaskNode(
            task_id=f"task_{int(time.time())}_001",
            task_type="compound",
            content=root_task
        )

        # Decompose into subtasks
        subtasks = self._generate_subtasks(goal, context)

        for i, subtask in enumerate(subtasks):
            subtask_node = TaskNode(
                task_id=f"task_{int(time.time())}_{i+2:03d}",
                task_type="compound" if isinstance(subtask, CompoundTask) else "primitive",
                content=subtask,
                parent=root_node
            )

            root_node.content.subtasks.append(subtask)
            self.task_registry[subtask_node.task_id] = subtask_node

        self.task_registry[root_node.task_id] = root_node
        return root_node

    def _generate_subtasks(self, goal: str, context: Dict[str, Any]) -> List[Any]:
        """Generate subtasks for a given goal"""
        # This is a simplified example - in reality, this would use LLM reasoning
        goal_lower = goal.lower()

        if "bring" in goal_lower and "water" in goal_lower:
            return [
                CompoundTask(
                    name="navigate_to_kitchen",
                    description="Go to the kitchen area",
                    subtasks=[
                        PrimitiveAction(
                            name="move_to",
                            parameters={"x": 2.0, "y": 3.0, "theta": 0.0},
                            duration=5.0,
                            preconditions=["robot_is_active"],
                            effects=["robot_at_kitchen"]
                        )
                    ]
                ),
                CompoundTask(
                    name="locate_water_source",
                    description="Find the water source",
                    subtasks=[
                        PrimitiveAction(
                            name="detect_object",
                            parameters={"type": "water_container"},
                            duration=3.0,
                            preconditions=["robot_at_kitchen"],
                            effects=["water_source_located"]
                        )
                    ]
                ),
                CompoundTask(
                    name="grasp_water_container",
                    description="Pick up the water container",
                    subtasks=[
                        PrimitiveAction(
                            name="approach_object",
                            parameters={"object_id": "water_container_001"},
                            duration=2.0,
                            preconditions=["water_source_located"],
                            effects=["robot_near_water_container"]
                        ),
                        PrimitiveAction(
                            name="grasp_object",
                            parameters={"object_id": "water_container_001"},
                            duration=4.0,
                            preconditions=["robot_near_water_container"],
                            effects=["water_container_grasped"]
                        )
                    ]
                ),
                CompoundTask(
                    name="navigate_to_user",
                    description="Return to the user",
                    subtasks=[
                        PrimitiveAction(
                            name="move_to",
                            parameters={"x": 0.0, "y": 0.0, "theta": 0.0},  # Assuming user at origin
                            duration=6.0,
                            preconditions=["water_container_grasped"],
                            effects=["robot_at_user"]
                        )
                    ]
                ),
                CompoundTask(
                    name="deliver_water",
                    description="Give the water to the user",
                    subtasks=[
                        PrimitiveAction(
                            name="release_object",
                            parameters={"object_id": "water_container_001"},
                            duration=2.0,
                            preconditions=["robot_at_user"],
                            effects=["water_delivered"]
                        )
                    ]
                )
            ]

        # Default fallback for unknown goals
        return [
            PrimitiveAction(
                name="speak",
                parameters={"text": "I'm not sure how to help with that"},
                duration=2.0,
                preconditions=[],
                effects=[]
            )
        ]

    def execute_plan(self, plan: TaskNode) -> bool:
        """Execute the hierarchical plan"""
        self.executor.start_execution()

        try:
            success = self._execute_task_node(plan)
            return success
        finally:
            self.executor.stop_execution()

    def _execute_task_node(self, node: TaskNode) -> bool:
        """Recursively execute a task node and its children"""
        if node.status == TaskStatus.CANCELLED:
            return False

        # Check dependencies
        if not self._check_dependencies(node):
            node.status = TaskStatus.FAILED
            return False

        if node.task_type == "primitive":
            # Execute primitive action
            action: PrimitiveAction = node.content

            # Check preconditions
            if not self._check_preconditions(action.preconditions):
                node.status = TaskStatus.FAILED
                return False

            # Execute the action
            success = self.executor.execute_primitive_action(action)

            if success:
                node.status = TaskStatus.SUCCESS
                self._apply_effects(action.effects)
            else:
                node.status = TaskStatus.FAILED

            return success

        elif node.task_type == "compound":
            # Execute compound task (sequence of subtasks)
            compound_task: CompoundTask = node.content
            node.status = TaskStatus.EXECUTING

            for subtask in compound_task.subtasks:
                # Convert subtask to TaskNode if needed
                if not isinstance(subtask, TaskNode):
                    subtask_node = TaskNode(
                        task_id=f"subtask_{int(time.time())}_{len(self.task_registry)}",
                        task_type="primitive" if isinstance(subtask, PrimitiveAction) else "compound",
                        content=subtask,
                        parent=node
                    )
                    self.task_registry[subtask_node.task_id] = subtask_node
                else:
                    subtask_node = subtask

                success = self._execute_task_node(subtask_node)
                if not success:
                    node.status = TaskStatus.FAILED
                    return False

            node.status = TaskStatus.SUCCESS
            return True

        return False

    def _check_dependencies(self, node: TaskNode) -> bool:
        """Check if all dependencies for a task are satisfied"""
        if not node.dependencies:
            return True

        for dep_id in node.dependencies:
            dep_node = self.task_registry.get(dep_id)
            if not dep_node or dep_node.status != TaskStatus.SUCCESS:
                return False

        return True

    def _check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if all preconditions are satisfied in the world state"""
        for condition in preconditions:
            if condition not in self.world_state or not self.world_state[condition]:
                return False
        return True

    def _apply_effects(self, effects: List[str]):
        """Apply effects to the world state"""
        for effect in effects:
            self.world_state[effect] = True

    def update_world_state(self, state_updates: Dict[str, Any]):
        """Update the world state with new information"""
        self.world_state.update(state_updates)

    def get_plan_status(self) -> Dict[str, Any]:
        """Get the status of the current plan"""
        if not self.task_tree:
            return {"status": "no_plan", "tasks": 0}

        def count_tasks(node: TaskNode) -> Tuple[int, int, int]:
            """Count total, completed, and failed tasks"""
            total = 1
            completed = 1 if node.status == TaskStatus.SUCCESS else 0
            failed = 1 if node.status == TaskStatus.FAILED else 0

            if node.task_type == "compound":
                compound_task: CompoundTask = node.content
                for subtask in compound_task.subtasks:
                    if isinstance(subtask, TaskNode):
                        sub_total, sub_completed, sub_failed = count_tasks(subtask)
                        total += sub_total
                        completed += sub_completed
                        failed += sub_failed

            return total, completed, failed

        total, completed, failed = count_tasks(self.task_tree)

        return {
            "status": self.task_tree.status.value,
            "total_tasks": total,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "completion_rate": completed / total if total > 0 else 0
        }

class TaskExecutor:
    """Execute primitive actions on the robot"""

    def __init__(self):
        self.is_executing = False
        self.current_action = None

    def start_execution(self):
        """Start the executor"""
        self.is_executing = True

    def stop_execution(self):
        """Stop the executor"""
        self.is_executing = False

    def execute_primitive_action(self, action: PrimitiveAction) -> bool:
        """Execute a primitive action"""
        self.current_action = action
        print(f"Executing action: {action.name} with params: {action.parameters}")

        # Simulate action execution
        success = self._simulate_action_execution(action)

        # Wait for action to complete (or timeout)
        time.sleep(min(action.duration, 10.0))  # Cap at 10 seconds

        self.current_action = None
        return success

    def _simulate_action_execution(self, action: PrimitiveAction) -> bool:
        """Simulate action execution"""
        # In a real implementation, this would interface with ROS 2 or robot hardware
        # For simulation, we'll return success for most actions
        print(f"Simulating {action.name} action...")

        # Simulate different action types
        if action.name == "move_to":
            x = action.parameters.get('x', 0)
            y = action.parameters.get('y', 0)
            print(f"  Moving to coordinates ({x}, {y})")
        elif action.name == "grasp_object":
            obj_id = action.parameters.get('object_id', 'unknown')
            print(f"  Grasping object {obj_id}")
        elif action.name == "speak":
            text = action.parameters.get('text', '')
            print(f"  Speaking: {text}")
        elif action.name == "detect_object":
            obj_type = action.parameters.get('type', 'unknown')
            print(f"  Detecting {obj_type}")

        # Simulate success rate based on action type
        import random
        success_rate = 0.95  # 95% success rate for simulation
        return random.random() < success_rate

def demonstrate_hierarchical_planning():
    """Demonstrate hierarchical task planning"""
    print("Demonstrating Hierarchical Task Planning")

    # Initialize planner
    planner = HierarchicalPlanner()

    # Example goal
    goal = "Bring me a cup of water from the kitchen"
    context = {
        "robot_position": [0.0, 0.0, 0.0],
        "kitchen_location": [2.0, 3.0, 0.0],
        "user_position": [0.0, 0.0, 0.0],
        "available_objects": ["cup", "water_bottle"]
    }

    print(f"\nGoal: {goal}")
    print(f"Context: {context}")

    # Create plan
    plan = planner.create_plan(goal, context)
    print(f"\nCreated hierarchical plan with root task: {plan.content.name}")

    # Show plan structure
    def print_plan_structure(node: TaskNode, depth: int = 0):
        indent = "  " * depth
        print(f"{indent}- {node.content.name} ({node.status.value})")

        if hasattr(node.content, 'subtasks'):
            for subtask in node.content.subtasks:
                if isinstance(subtask, TaskNode):
                    print_plan_structure(subtask, depth + 1)
                else:
                    print(f"{indent}  - {subtask.name if hasattr(subtask, 'name') else 'Action'}")

    print(f"\nPlan structure:")
    print_plan_structure(plan)

    # Execute plan
    print(f"\nExecuting plan...")
    success = planner.execute_plan(plan)

    print(f"Plan execution {'succeeded' if success else 'failed'}")
    print(f"Final plan status: {planner.get_plan_status()}")

if __name__ == "__main__":
    demonstrate_hierarchical_planning()
```

## ROS 2 Action Integration

### Action Server Implementation

```python
# python/ros2_action_integration.py
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile

# Import standard ROS 2 action interfaces
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

# Custom action interfaces would be defined here
# For this example, we'll define a generic cognitive action interface

class LLMCognitiveActionServer(Node):
    """ROS 2 action server for LLM-based cognitive planning"""

    def __init__(self):
        super().__init__('llm_cognitive_action_server')

        # Initialize LLM interface
        self.llm_interface = LLMRobotInterface(
            api_key="YOUR_API_KEY",  # This would be passed securely
            model="gpt-4-turbo"
        )

        # Initialize context manager
        self.context_manager = ContextManager()
        self.hierarchical_planner = HierarchicalPlanner()

        # Setup action server
        self._action_server = ActionServer(
            self,
            # Define custom action type - in practice, you'd generate this with action interface
            # For now, we'll simulate with a generic structure
            'llm_cognitive_action',
            'ExecuteCognitiveTask',  # This would be your custom action
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Publishers for state updates
        self.state_publisher = self.create_publisher(String, 'cognitive_state', 10)
        self.plan_publisher = self.create_publisher(String, 'execution_plan', 10)

        # Subscribers for context updates
        self.context_subscriber = self.create_subscription(
            String,
            'context_update',
            self.context_callback,
            10
        )

        self.get_logger().info("LLM Cognitive Action Server initialized")

    def goal_callback(self, goal_request):
        """Accept or reject goal requests"""
        self.get_logger().info(f"Received cognitive task goal: {goal_request.task_description}")

        # Validate the goal
        if self._validate_goal(goal_request):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info("Received cancel request for cognitive task")
        return CancelResponse.ACCEPT

    def context_callback(self, msg):
        """Handle context updates"""
        try:
            context_data = json.loads(msg.data)
            self.update_context(context_data)
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid context JSON: {msg.data}")

    def update_context(self, context_data: Dict[str, Any]):
        """Update context from ROS 2 messages"""
        # Update context based on message content
        if 'location' in context_data:
            self.context_manager.add_context(
                ContextType.SPATIAL,
                context_data['location'],
                confidence=0.9
            )

        if 'object' in context_data:
            self.context_manager.add_context(
                ContextType.OBJECT,
                context_data['object'],
                confidence=0.8
            )

    def _validate_goal(self, goal_request) -> bool:
        """Validate the cognitive task goal"""
        # Check if the goal is well-formed
        if not hasattr(goal_request, 'task_description') or not goal_request.task_description:
            return False

        # Check if the robot is capable of the requested task
        # This would involve checking robot capabilities vs. goal requirements
        return True

    async def execute_callback(self, goal_handle):
        """Execute the cognitive task"""
        self.get_logger().info("Executing cognitive task...")

        feedback_msg = None  # Define feedback message type
        result = None  # Define result message type

        try:
            # Get the goal
            goal = goal_handle.request
            task_description = goal.task_description

            # Process the natural language command
            cognitive_task = await self.llm_interface.process_natural_language(
                task_description,
                context=self._get_current_context()
            )

            # Publish the plan
            plan_msg = String()
            plan_msg.data = json.dumps({
                'task_id': cognitive_task.id,
                'actions': [action.action_type for action in cognitive_task.actions]
            })
            self.plan_publisher.publish(plan_msg)

            # Create and execute hierarchical plan
            plan = self.hierarchical_planner.create_plan(
                goal=task_description,
                context=self._get_current_context()
            )

            # Execute the plan
            success = self.hierarchical_planner.execute_plan(plan)

            # Update result based on execution
            if success:
                goal_handle.succeed()
                # result = YourResultType(success=True, message="Task completed successfully")
            else:
                goal_handle.abort()
                # result = YourResultType(success=False, message="Task execution failed")

        except Exception as e:
            self.get_logger().error(f"Error executing cognitive task: {e}")
            goal_handle.abort()
            # result = YourResultType(success=False, message=f"Error: {str(e)}")

        return result

    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context for LLM processing"""
        # Gather context from various sources
        context = {
            'spatial': self.context_manager.get_spatial_context(),
            'objects': self.context_manager.get_object_context(),
            'recent_actions': self.context_manager.get_recent_context(time_window=300),  # 5 minutes
            'robot_state': self.llm_interface.robot_state,
            'environment': self.llm_interface.environment_map
        }
        return context

class LLMClientInterface(Node):
    """Client interface for sending cognitive tasks to the action server"""

    def __init__(self):
        super().__init__('llm_client_interface')

        # Create action client
        self._action_client = rclpy.action.ActionClient(
            self,
            # Your custom action type here
            'ExecuteCognitiveTask',
            'llm_cognitive_action'
        )

        # Publisher for natural language commands
        self.command_publisher = self.create_publisher(String, 'natural_language_command', 10)

        # Subscriber for results
        self.result_subscriber = self.create_subscription(
            String,
            'cognitive_result',
            self.result_callback,
            10
        )

        self.get_logger().info("LLM Client Interface initialized")

    def send_cognitive_task(self, command: str) -> bool:
        """Send a cognitive task to the action server"""
        goal_msg = None  # Your goal message type

        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available")
            return False

        # Send goal
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        # Wait for result
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected by server")
            return False

        self.get_logger().info("Goal accepted by server, waiting for result...")

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result().result
        self.get_logger().info(f"Result: {result}")

        return True

    def feedback_callback(self, feedback_msg):
        """Handle feedback from action server"""
        self.get_logger().info(f"Received feedback: {feedback_msg}")

    def result_callback(self, msg):
        """Handle result messages"""
        self.get_logger().info(f"Cognitive result: {msg.data}")

def main(args=None):
    """Main function to run the LLM-ROS2 integration"""
    rclpy.init(args=args)

    # Create nodes
    server_node = LLMCognitiveActionServer()
    client_node = LLMClientInterface()

    # Use multi-threaded executor to handle both nodes
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(server_node)
    executor.add_node(client_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Safety and Error Handling

### Robust Execution Framework

```python
# python/safety_framework.py
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import traceback
from contextlib import contextmanager

class SafetyLevel(Enum):
    """Safety levels for robot actions"""
    HIGHEST = 4  # Emergency stop level
    HIGH = 3     # Safety-critical actions
    MEDIUM = 2   # Normal operations
    LOW = 1      # Low-risk operations

class SafetyViolation(Exception):
    """Exception raised when a safety violation occurs"""
    def __init__(self, message: str, violation_type: str = "unknown"):
        super().__init__(message)
        self.violation_type = violation_type

class SafetyMonitor:
    """Monitors robot state and prevents unsafe actions"""

    def __init__(self):
        self.safety_level = SafetyLevel.MEDIUM
        self.is_emergency = False
        self.violation_history = []
        self.constraints = []
        self.callbacks = {
            'violation': [],
            'recovery': [],
            'state_change': []
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def add_constraint(self, constraint_func: Callable[[], bool], description: str):
        """Add a safety constraint function"""
        self.constraints.append({
            'function': constraint_func,
            'description': description
        })

    def check_safety(self, action: RobotAction) -> bool:
        """Check if an action is safe to execute"""
        if self.is_emergency:
            raise SafetyViolation("Emergency stop active", "emergency")

        # Check safety level constraints
        action_safety_level = self._get_action_safety_level(action)
        if action_safety_level.value > self.safety_level.value:
            raise SafetyViolation(
                f"Action safety level {action_safety_level.name} exceeds current safety level {self.safety_level.name}",
                "level_mismatch"
            )

        # Check all constraints
        for constraint in self.constraints:
            try:
                if not constraint['function']():
                    raise SafetyViolation(
                        f"Constraint violated: {constraint['description']}",
                        "constraint_violation"
                    )
            except Exception as e:
                raise SafetyViolation(f"Constraint check failed: {str(e)}", "constraint_error")

        return True

    def _get_action_safety_level(self, action: RobotAction) -> SafetyLevel:
        """Determine safety level for an action"""
        high_risk_actions = ['move_to', 'approach_object', 'grasp_object', 'manipulate']
        medium_risk_actions = ['speak', 'gesture', 'detect_object']

        if action.action_type in high_risk_actions:
            return SafetyLevel.HIGH
        elif action.action_type in medium_risk_actions:
            return SafetyLevel.MEDIUM
        else:
            return SafetyLevel.LOW

    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.is_emergency = True
        self.safety_level = SafetyLevel.HIGHEST
        self.logger.warning("Emergency stop triggered!")

        # Call emergency callbacks
        for callback in self.callbacks['violation']:
            try:
                callback("EMERGENCY_STOP", "Emergency stop activated")
            except Exception as e:
                self.logger.error(f"Error in emergency callback: {e}")

    def clear_emergency(self):
        """Clear emergency state"""
        self.is_emergency = False
        self.safety_level = SafetyLevel.MEDIUM
        self.logger.info("Emergency cleared, returning to normal operations")

        # Call recovery callbacks
        for callback in self.callbacks['recovery']:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in recovery callback: {e}")

    def register_callback(self, event_type: str, callback: Callable):
        """Register a callback for safety events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)

    def log_violation(self, violation: SafetyViolation):
        """Log a safety violation"""
        violation_entry = {
            'timestamp': time.time(),
            'violation': str(violation),
            'type': violation.violation_type,
            'traceback': traceback.format_exc()
        }
        self.violation_history.append(violation_entry)

        self.logger.error(f"Safety violation: {violation_entry}")

class RobustActionExecutor:
    """Robust executor with safety and error handling"""

    def __init__(self, safety_monitor: SafetyMonitor):
        self.safety_monitor = safety_monitor
        self.is_running = False
        self.executor_thread = None
        self.action_queue = asyncio.Queue()
        self.active_action = None

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def start_execution(self):
        """Start the robust execution loop"""
        self.is_running = True
        self.executor_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.executor_thread.start()

    def stop_execution(self):
        """Stop the execution loop"""
        self.is_running = False
        if self.executor_thread:
            self.executor_thread.join()

    def queue_action(self, action: RobotAction):
        """Queue an action for execution"""
        asyncio.run_coroutine_threadsafe(
            self.action_queue.put(action),
            asyncio.get_event_loop()
        )

    def _execution_loop(self):
        """Main execution loop with safety checks"""
        while self.is_running:
            try:
                # Get next action from queue
                action = asyncio.run_coroutine_threadsafe(
                    self.action_queue.get(),
                    asyncio.get_event_loop()
                ).result(timeout=0.1)

                if action:
                    self._execute_action_with_safety(action)

            except asyncio.TimeoutError:
                continue  # No action available, continue loop
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(0.1)

    def _execute_action_with_safety(self, action: RobotAction):
        """Execute an action with comprehensive safety checks"""
        self.active_action = action

        try:
            # Check safety before execution
            self.safety_monitor.check_safety(action)

            # Log action start
            self.logger.info(f"Starting action: {action.action_type} with params: {action.parameters}")

            # Execute action with timeout
            success = self._execute_with_timeout(action)

            if success:
                self.logger.info(f"Action completed successfully: {action.action_type}")
            else:
                self.logger.warning(f"Action failed: {action.action_type}")

        except SafetyViolation as sv:
            self.logger.error(f"Safety violation during action {action.action_type}: {sv}")
            self.safety_monitor.log_violation(sv)

            # Trigger appropriate response based on violation type
            if sv.violation_type == "emergency":
                self.safety_monitor.trigger_emergency_stop()

        except Exception as e:
            self.logger.error(f"Unexpected error executing action {action.action_type}: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            self.active_action = None

    def _execute_with_timeout(self, action: RobotAction, timeout: float = 30.0) -> bool:
        """Execute action with timeout protection"""
        start_time = time.time()

        try:
            # Simulate action execution with timeout
            # In a real implementation, this would interface with the robot
            success = self._simulate_action_execution(action)

            execution_time = time.time() - start_time
            if execution_time > timeout:
                self.logger.warning(f"Action exceeded timeout: {action.action_type}")
                return False

            return success

        except Exception as e:
            self.logger.error(f"Error during action execution: {e}")
            return False

    def _simulate_action_execution(self, action: RobotAction) -> bool:
        """Simulate action execution with safety checks"""
        # In a real implementation, this would interface with ROS 2 or robot hardware
        # For simulation, we'll include safety checks

        # Simulate different action types with potential safety issues
        if action.action_type == "move_to":
            # Check for collision risks
            x = action.parameters.get('x', 0)
            y = action.parameters.get('y', 0)

            # Simulate potential collision detection
            if self._would_collide(x, y):
                raise SafetyViolation(f"Collision risk at coordinates ({x}, {y})", "collision_risk")

        elif action.action_type == "grasp_object":
            obj_id = action.parameters.get('object_id', 'unknown')
            # Check if object is safe to grasp
            if not self._is_safe_to_grasp(obj_id):
                raise SafetyViolation(f"Object {obj_id} is not safe to grasp", "unsafe_grasp")

        # Simulate successful execution
        time.sleep(min(action.parameters.get('duration', 1.0), 5.0))
        return True

    def _would_collide(self, x: float, y: float) -> bool:
        """Simulate collision detection"""
        # In a real implementation, this would check against map or sensor data
        # For simulation, return True for certain coordinates
        return abs(x) > 10.0 or abs(y) > 10.0  # Simulate boundary collision

    def _is_safe_to_grasp(self, obj_id: str) -> bool:
        """Check if an object is safe to grasp"""
        # In a real implementation, this would check object properties
        # For simulation, return False for certain object IDs
        dangerous_objects = ["hot_item", "fragile_item", "sharp_object"]
        return obj_id not in dangerous_objects

class RecoveryManager:
    """Manages recovery from errors and safety violations"""

    def __init__(self, safety_monitor: SafetyMonitor, action_executor: RobustActionExecutor):
        self.safety_monitor = safety_monitor
        self.action_executor = action_executor
        self.recovery_strategies = {}
        self.failed_actions = []

        # Register default recovery strategies
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default recovery strategies"""
        self.recovery_strategies = {
            'collision_risk': self._recover_from_collision_risk,
            'constraint_violation': self._recover_from_constraint_violation,
            'timeout': self._recover_from_timeout,
            'emergency': self._recover_from_emergency
        }

    def handle_violation(self, violation: SafetyViolation, action: RobotAction):
        """Handle a safety violation and attempt recovery"""
        self.failed_actions.append({
            'action': action,
            'violation': violation,
            'timestamp': time.time()
        })

        strategy = self.recovery_strategies.get(violation.violation_type)
        if strategy:
            try:
                return strategy(violation, action)
            except Exception as e:
                self.action_executor.logger.error(f"Recovery strategy failed: {e}")
                return False
        else:
            self.action_executor.logger.warning(f"No recovery strategy for violation type: {violation.violation_type}")
            return False

    def _recover_from_collision_risk(self, violation: SafetyViolation, action: RobotAction) -> bool:
        """Recovery strategy for collision risks"""
        self.action_executor.logger.info("Attempting collision risk recovery...")

        # Stop current movement
        stop_action = RobotAction(
            action_type="stop_immediately",
            parameters={}
        )

        # Try to execute stop action
        try:
            self.action_executor._execute_with_timeout(stop_action, timeout=2.0)
        except:
            pass  # Stop action might fail if already stopped

        # Plan alternative route (simplified)
        self.action_executor.logger.info("Collision risk recovery completed")
        return True

    def _recover_from_constraint_violation(self, violation: SafetyViolation, action: RobotAction) -> bool:
        """Recovery strategy for constraint violations"""
        self.action_executor.logger.info("Attempting constraint violation recovery...")

        # Log the constraint that was violated
        self.action_executor.logger.info(f"Constraint violation: {violation}")

        # In a real system, you might relax constraints or modify the action
        # For now, just return True to indicate recovery was attempted
        return True

    def _recover_from_timeout(self, violation: SafetyViolation, action: RobotAction) -> bool:
        """Recovery strategy for timeouts"""
        self.action_executor.logger.info("Attempting timeout recovery...")

        # Try to cancel the timed-out action
        stop_action = RobotAction(
            action_type="stop_immediately",
            parameters={}
        )

        try:
            self.action_executor._execute_with_timeout(stop_action, timeout=2.0)
        except:
            pass

        return True

    def _recover_from_emergency(self, violation: SafetyViolation, action: RobotAction) -> bool:
        """Recovery strategy for emergency situations"""
        self.action_executor.logger.info("Attempting emergency recovery...")

        # Wait for emergency to clear
        start_time = time.time()
        while self.safety_monitor.is_emergency and (time.time() - start_time) < 10.0:
            time.sleep(0.1)

        if not self.safety_monitor.is_emergency:
            self.action_executor.logger.info("Emergency cleared, recovery successful")
            return True
        else:
            self.action_executor.logger.error("Emergency did not clear within timeout")
            return False

def demonstrate_safety_framework():
    """Demonstrate the safety framework"""
    print("Demonstrating Safety Framework for LLM-Based Robotics")

    # Initialize safety monitor
    safety_monitor = SafetyMonitor()

    # Add some safety constraints
    def workspace_boundary_constraint():
        """Constraint: robot must stay within workspace boundaries"""
        # Simulate checking robot position
        robot_x, robot_y = 5.0, 3.0  # Simulated position
        return abs(robot_x) <= 10.0 and abs(robot_y) <= 10.0

    def no_dangerous_objects_constraint():
        """Constraint: no dangerous objects in workspace"""
        # Simulate checking for dangerous objects
        dangerous_objects_detected = False  # Simulated detection
        return not dangerous_objects_detected

    safety_monitor.add_constraint(workspace_boundary_constraint, "Workspace boundary check")
    safety_monitor.add_constraint(no_dangerous_objects_constraint, "Dangerous object check")

    # Initialize robust executor
    executor = RobustActionExecutor(safety_monitor)

    # Initialize recovery manager
    recovery_manager = RecoveryManager(safety_monitor, executor)

    # Register violation callback
    def on_violation(violation_type, message):
        print(f"Safety violation: {violation_type} - {message}")
        # Attempt recovery
        recovery_manager.handle_violation(
            SafetyViolation(message, violation_type),
            RobotAction("test_action", {})
        )

    safety_monitor.register_callback('violation', on_violation)

    # Test safe action
    print("\nTesting safe action...")
    safe_action = RobotAction(
        action_type="speak",
        parameters={"text": "Hello, I am operating safely"}
    )
    executor.queue_action(safe_action)

    # Test potentially unsafe action (collision risk)
    print("\nTesting potentially unsafe action...")
    risky_action = RobotAction(
        action_type="move_to",
        parameters={"x": 15.0, "y": 15.0, "theta": 0.0}  # Outside boundary
    )
    executor.queue_action(risky_action)

    # Start execution
    executor.start_execution()

    # Let it run for a bit
    time.sleep(3)

    # Stop execution
    executor.stop_execution()

    print(f"\nSafety violations logged: {len(safety_monitor.violation_history)}")
    print("Safety framework demonstration completed.")

if __name__ == "__main__":
    demonstrate_safety_framework()
```

## Performance Optimization

### Optimized LLM Integration

```python
# python/optimized_integration.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from typing import Dict, List, Optional, Any, Callable
import openai
from functools import lru_cache
import gc
import psutil
import os

class OptimizedLLMInterface:
    """Optimized interface for LLM-robotics integration"""

    def __init__(self, api_key: str, model: str = "gpt-4-turbo", max_workers: int = 2):
        openai.api_key = api_key
        self.model = model
        self.max_workers = max_workers

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Caching for frequently used responses
        self.response_cache = {}
        self.cache_size_limit = 100

        # Rate limiting
        self.request_times = []
        self.max_requests_per_minute = 30  # Adjust based on your API plan

        # Statistics
        self.stats = {
            'total_requests': 0,
            'cached_responses': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0
        }

        # Setup async event loop
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def process_request(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a request with optimization"""
        start_time = time.time()

        # Create cache key
        cache_key = self._create_cache_key(prompt, context)

        # Check cache first
        if cache_key in self.cache:
            self.stats['cached_responses'] += 1
            cached_response = self.cache[cache_key]
            self.stats['average_response_time'] = (
                (self.stats['average_response_time'] * (self.stats['total_requests']) + (time.time() - start_time)) /
                (self.stats['total_requests'] + 1)
            )
            self.stats['cache_hit_rate'] = self.stats['cached_responses'] / (self.stats['total_requests'] + 1)
            return cached_response

        # Apply rate limiting
        self._enforce_rate_limit()

        # Make API call
        try:
            response = self._make_api_call(prompt, context)
        except Exception as e:
            # Return a safe fallback response
            response = {"actions": [{"type": "speak", "parameters": {"text": f"Error processing request: {str(e)}"}}]}

        # Update statistics
        response_time = time.time() - start_time
        self.stats['total_requests'] += 1
        self.stats['average_response_time'] = (
            (self.stats['average_response_time'] * (self.stats['total_requests'] - 1) + response_time) /
            self.stats['total_requests']
        )
        self.stats['cache_hit_rate'] = self.stats['cached_responses'] / self.stats['total_requests']

        # Add to cache
        self._add_to_cache(cache_key, response)

        return response

    def _create_cache_key(self, prompt: str, context: Dict[str, Any]) -> str:
        """Create a cache key from prompt and context"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        return f"{prompt[:100]}_{hash(context_str)}"  # Limit prompt length in key

    def _enforce_rate_limit(self):
        """Enforce API rate limiting"""
        current_time = time.time()

        # Remove old requests outside the minute window
        self.request_times = [req_time for req_time in self.request_times
                            if current_time - req_time < 60]

        # Check if we're over the limit
        if len(self.request_times) >= self.max_requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Add current request time
        self.request_times.append(current_time)

    def _make_api_call(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual API call with error handling"""
        system_message = (
            "You are a cognitive planning assistant for a humanoid robot. "
            "Your job is to interpret natural language commands and generate "
            "executable action plans. Always respond with valid JSON containing "
            "an 'actions' array. Each action should have 'type' and 'parameters'."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # Add context if provided
        if context:
            messages.append({"role": "user", "content": f"Context: {json.dumps(context)}"})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            timeout=30
        )

        content = response.choices[0].message.content.strip()

        # Clean up response
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content)

    def _add_to_cache(self, key: str, response: Dict[str, Any]):
        """Add response to cache with size management"""
        if len(self.cache) >= self.cache_size_limit:
            # Remove oldest entries
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = response

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats.copy()

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        self.stats['cached_responses'] = 0

class BatchProcessor:
    """Process multiple requests in batches for efficiency"""

    def __init__(self, llm_interface: OptimizedLLMInterface, batch_size: int = 5):
        self.llm_interface = llm_interface
        self.batch_size = batch_size
        self.request_queue = []
        self.processing_lock = threading.Lock()

    def add_request(self, prompt: str, context: Dict[str, Any] = None) -> asyncio.Future:
        """Add a request to the batch queue"""
        future = asyncio.run_coroutine_threadsafe(
            self._process_single_request(prompt, context),
            self.llm_interface.loop
        )
        return future

    async def _process_single_request(self, prompt: str, context: Dict[str, Any]):
        """Process a single request"""
        return self.llm_interface.process_request(prompt, context)

    def process_batch(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of requests"""
        results = []

        # Process in chunks of batch_size
        for i in range(0, len(requests), self.batch_size):
            chunk = requests[i:i + self.batch_size]

            # Submit all requests in the chunk
            futures = []
            for req in chunk:
                future = self.add_request(req['prompt'], req.get('context'))
                futures.append(future)

            # Wait for all to complete
            for future in futures:
                result = future.result()  # This will block until complete
                results.append(result)

        return results

class MemoryOptimizer:
    """Optimize memory usage for LLM integration"""

    def __init__(self, memory_limit_mb: int = 1024):
        self.memory_limit_mb = memory_limit_mb
        self.current_memory_usage = 0

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb <= self.memory_limit_mb

    def trigger_garbage_collection(self):
        """Trigger garbage collection to free memory"""
        collected = gc.collect()
        print(f"Garbage collected {collected} objects")

    def optimize_context_size(self, context: Dict[str, Any], max_size: int = 5000) -> Dict[str, Any]:
        """Optimize context size to reduce memory usage"""
        context_str = json.dumps(context)

        if len(context_str) > max_size:
            # Truncate context while preserving important information
            truncated_context = self._truncate_context(context, max_size)
            return truncated_context

        return context

    def _truncate_context(self, context: Dict[str, Any], max_size: int) -> Dict[str, Any]:
        """Truncate context while preserving important parts"""
        # Keep recent and important context, truncate older parts
        truncated = {}

        for key, value in context.items():
            if isinstance(value, list) and len(value) > 10:  # Truncate long lists
                truncated[key] = value[-10:]  # Keep last 10 items
            elif isinstance(value, str) and len(value) > 1000:  # Truncate long strings
                truncated[key] = value[:1000] + "... (truncated)"
            else:
                truncated[key] = value

        return truncated

def benchmark_optimized_interface():
    """Benchmark the optimized interface"""
    print("Benchmarking Optimized LLM Interface...")

    # Initialize with a placeholder API key
    # In practice, you would use your actual API key
    try:
        interface = OptimizedLLMInterface(
            api_key="YOUR_API_KEY",
            model="gpt-3.5-turbo",  # Use smaller model for testing
            max_workers=2
        )

        # Test prompts
        test_prompts = [
            "Move to the kitchen and bring me a cup",
            "Find the red ball and pick it up",
            "Go to the living room and greet the person there",
            "Locate the book on the table and move it to the shelf",
            "Turn off the lights in the bedroom"
        ]

        print(f"\nProcessing {len(test_prompts)} prompts...")

        start_time = time.time()
        for i, prompt in enumerate(test_prompts):
            print(f"Processing {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            result = interface.process_request(prompt)
            print(f"  Response: {len(str(result))} chars")

        total_time = time.time() - start_time
        stats = interface.get_stats()

        print(f"\nBenchmark Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests processed: {stats['total_requests']}")
        print(f"  Average response time: {stats['average_response_time']:.3f}s")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"  Cached responses: {stats['cached_responses']}")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        print("Make sure you have a valid API key and internet connection")

if __name__ == "__main__":
    benchmark_optimized_interface()
```

## Best Practices for LLM-Robotics Integration

### Design Guidelines

1. **Context Management**: Maintain conversation and task context for coherent interaction
2. **Safety First**: Implement robust safety checks and fallback mechanisms
3. **Error Handling**: Plan for LLM failures and ambiguous outputs
4. **Latency Optimization**: Balance response quality with real-time requirements
5. **Privacy Considerations**: Handle sensitive user data appropriately

### Performance Considerations

- **API Costs**: Monitor and optimize API usage to manage costs
- **Response Time**: Implement caching and batching for common requests
- **Memory Usage**: Optimize context size and implement memory management
- **Reliability**: Implement fallback mechanisms for API failures
- **Scalability**: Design systems that can handle multiple concurrent users

## Hands-On Exercise

### Exercise: Building an LLM-Powered Robot Assistant

1. **Setup LLM Environment**
   - Configure OpenAI API access
   - Implement basic LLM interface

2. **Create Context Manager**
   - Implement context tracking for spatial, object, and temporal information
   - Add context resolution for ambiguous references

3. **Develop Hierarchical Planner**
   - Create task decomposition system
   - Implement action execution with safety checks

4. **Integrate with ROS 2**
   - Create action server for cognitive tasks
   - Implement client interface for command submission

5. **Test and Optimize**
   - Test with various natural language commands
   - Optimize for response time and accuracy
   - Validate safety and error handling

## Summary

LLM-based cognitive planning enables natural language interaction with humanoid robots by bridging high-level commands with low-level robot actions. The key components include natural language understanding with context awareness, hierarchical task planning, and safe action execution. Proper implementation requires careful attention to safety, error handling, and performance optimization. By following best practices for context management, safety checks, and API optimization, we can create robust and responsive LLM-powered robotic systems.

## Learning Path Adjustment

Based on your experience level, you may want to focus on:

- **Beginner**: Focus on basic LLM integration and simple command processing
- **Intermediate**: Dive deeper into context management and hierarchical planning
- **Advanced**: Explore custom model fine-tuning and complex multi-modal integration