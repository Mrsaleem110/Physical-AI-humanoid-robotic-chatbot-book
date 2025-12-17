from typing import Any, Dict, List
from .base_agent import BaseAgent, AgentType, AgentSkill
import asyncio
import json
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    MOVE_TO = "move_to"
    GRASP = "grasp"
    PLACE = "place"
    OPEN = "open"
    CLOSE = "close"
    PUSH = "push"
    PULL = "pull"
    FOLLOW = "follow"
    STOP = "stop"


@dataclass
class Action:
    type: ActionType
    parameters: Dict[str, Any]
    priority: int = 1
    estimated_duration: float = 1.0  # in seconds


class VLAActionPlanningAgent(BaseAgent):
    """
    VLA (Vision-Language-Action) Action Planning Subagent
    Handles vision-language-action pipeline and cognitive planning
    """

    def __init__(self):
        super().__init__(
            agent_type=AgentType.VLA,
            name="VLA Action Planning Agent",
            description="Specialized in vision-language-action pipeline and cognitive planning for robotics"
        )
        # Add relevant skills
        self.add_skill(AgentSkill.PLANNING)
        self.add_skill(AgentSkill.REASONING)
        self.add_skill(AgentSkill.EXECUTION)
        self.add_skill(AgentSkill.COMMUNICATION)

        # Initialize VLA components
        self.vision_processor = None
        self.language_model = None
        self.action_planner = None
        self.object_detector = None
        self.manipulation_planner = None

        # Action execution state
        self.current_plan = []
        self.executing_action = None
        self.action_history = []

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute VLA tasks
        """
        task_type = task.get("type", "plan_action")
        parameters = task.get("parameters", {})

        if task_type == "natural_language_to_action":
            return await self._process_natural_language_to_action(parameters)
        elif task_type == "visual_perception":
            return await self._process_visual_perception(parameters)
        elif task_type == "action_planning":
            return await self._create_action_plan(parameters)
        elif task_type == "action_execution":
            return await self._execute_action_plan(parameters)
        elif task_type == "object_detection":
            return await self._detect_objects(parameters)
        elif task_type == "manipulation_planning":
            return await self._plan_manipulation(parameters)
        else:
            return await self._process_general_task(parameters)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """
        Determine if this agent can handle the given task
        """
        task_type = task.get("type", "")
        intent = task.get("intent", "")
        required_skills = task.get("required_skills", [])

        # Check if task type matches agent capabilities
        vla_related = any(keyword in task_type.lower() for keyword in
                         ["vision", "language", "action", "plan", "perception", "manipulation", "cognitive"])

        # Check if intent contains action-related keywords
        action_related = any(keyword in intent.lower() for keyword in
                            ["move", "grasp", "place", "pick", "put", "open", "close", "push", "pull", "follow"])

        # Check if required skills are supported
        required_skills_supported = all(
            skill in [s.value for s in self.skills] for skill in required_skills
        )

        return vla_related or action_related or required_skills_supported

    async def _process_natural_language_to_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process natural language to robot actions
        """
        text_input = parameters.get("text", "")
        context = parameters.get("context", {})
        priority = parameters.get("priority", 1)

        # Parse natural language and create action plan
        parsed_intent = await self._parse_natural_language(text_input)
        action_plan = await self._create_action_plan_from_intent(parsed_intent, context)

        return {
            "input_text": text_input,
            "parsed_intent": parsed_intent,
            "action_plan": action_plan,
            "status": "success"
        }

    async def _parse_natural_language(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language to extract intent and entities
        """
        # This is a simplified implementation
        # In a real system, this would use NLP models to extract intent and entities
        text_lower = text.lower()

        # Extract basic intent and entities
        intent = "unknown"
        entities = []

        if any(word in text_lower for word in ["move", "go to", "navigate to"]):
            intent = "navigation"
            entities.extend(self._extract_location_entities(text))
        elif any(word in text_lower for word in ["grasp", "pick", "take", "grab"]):
            intent = "grasping"
            entities.extend(self._extract_object_entities(text))
        elif any(word in text_lower for word in ["place", "put", "set"]):
            intent = "placement"
            entities.extend(self._extract_object_entities(text))
            entities.extend(self._extract_location_entities(text))
        elif any(word in text_lower for word in ["open", "close"]):
            intent = "manipulation"
            entities.extend(self._extract_object_entities(text))

        return {
            "intent": intent,
            "entities": entities,
            "original_text": text,
            "confidence": 0.9  # Simulated confidence
        }

    def _extract_location_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract location entities from text
        """
        # Simplified location extraction
        locations = ["kitchen", "living room", "bedroom", "office", "table", "shelf", "cabinet"]
        found_locations = []

        for loc in locations:
            if loc in text.lower():
                found_locations.append({
                    "type": "location",
                    "value": loc,
                    "confidence": 0.8
                })

        return found_locations

    def _extract_object_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract object entities from text
        """
        # Simplified object extraction
        objects = ["cup", "book", "bottle", "box", "apple", "phone", "keys", "pen"]
        found_objects = []

        for obj in objects:
            if obj in text.lower():
                found_objects.append({
                    "type": "object",
                    "value": obj,
                    "confidence": 0.8
                })

        return found_objects

    async def _create_action_plan_from_intent(self, parsed_intent: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create action plan from parsed intent and context
        """
        intent = parsed_intent.get("intent", "unknown")
        entities = parsed_intent.get("entities", [])

        actions = []

        if intent == "navigation":
            # Find location entity
            location = next((e for e in entities if e["type"] == "location"), None)
            if location:
                actions.append({
                    "type": ActionType.MOVE_TO,
                    "parameters": {"target_location": location["value"]},
                    "priority": 1
                })

        elif intent == "grasping":
            # Find object entity
            obj = next((e for e in entities if e["type"] == "object"), None)
            if obj:
                actions.append({
                    "type": ActionType.MOVE_TO,
                    "parameters": {"target_object": obj["value"]},
                    "priority": 1
                })
                actions.append({
                    "type": ActionType.GRASP,
                    "parameters": {"object": obj["value"]},
                    "priority": 2
                })

        elif intent == "placement":
            # Find object and location entities
            obj = next((e for e in entities if e["type"] == "object"), None)
            location = next((e for e in entities if e["type"] == "location"), None)

            if obj and location:
                actions.append({
                    "type": ActionType.MOVE_TO,
                    "parameters": {"target_object": obj["value"]},
                    "priority": 1
                })
                actions.append({
                    "type": ActionType.GRASP,
                    "parameters": {"object": obj["value"]},
                    "priority": 2
                })
                actions.append({
                    "type": ActionType.MOVE_TO,
                    "parameters": {"target_location": location["value"]},
                    "priority": 3
                })
                actions.append({
                    "type": ActionType.PLACE,
                    "parameters": {"object": obj["value"], "location": location["value"]},
                    "priority": 4
                })

        elif intent == "manipulation":
            obj = next((e for e in entities if e["type"] == "object"), None)
            if obj:
                action_type = ActionType.OPEN if "open" in parsed_intent.get("original_text", "").lower() else ActionType.CLOSE
                actions.append({
                    "type": action_type,
                    "parameters": {"object": obj["value"]},
                    "priority": 1
                })

        return actions

    async def _create_action_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an action plan based on goals and constraints
        """
        goals = parameters.get("goals", [])
        constraints = parameters.get("constraints", {})
        environment_state = parameters.get("environment_state", {})

        # In a real implementation, this would use sophisticated planning algorithms
        # For simulation, we'll create a simple plan based on goals

        plan = []
        for goal in goals:
            if goal.get("type") == "navigation":
                plan.append({
                    "action": ActionType.MOVE_TO,
                    "parameters": goal.get("parameters", {}),
                    "estimated_duration": 5.0
                })
            elif goal.get("type") == "manipulation":
                plan.append({
                    "action": ActionType.GRASP,
                    "parameters": goal.get("parameters", {}),
                    "estimated_duration": 3.0
                })

        self.current_plan = plan

        return {
            "plan": plan,
            "total_actions": len(plan),
            "estimated_duration": sum(a.get("estimated_duration", 1.0) for a in plan),
            "status": "created"
        }

    async def _execute_action_plan(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action plan
        """
        plan = parameters.get("plan", self.current_plan)
        execution_context = parameters.get("context", {})

        results = []
        for action in plan:
            result = await self._execute_single_action(action, execution_context)
            results.append(result)

            # Check if execution should be interrupted
            if result.get("status") == "error":
                break

        return {
            "executed_actions": results,
            "completed_count": len([r for r in results if r.get("status") == "success"]),
            "total_count": len(results),
            "status": "completed" if all(r.get("status") == "success" for r in results) else "partial"
        }

    async def _execute_single_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action
        """
        action_type = action.get("action", "")
        parameters = action.get("parameters", {})

        # Simulate action execution
        await asyncio.sleep(action.get("estimated_duration", 1.0))

        # Record in action history
        action_record = {
            "action": action_type,
            "parameters": parameters,
            "status": "success",
            "execution_time": action.get("estimated_duration", 1.0),
            "timestamp": asyncio.get_event_loop().time()
        }

        self.action_history.append(action_record)

        return action_record

    async def _process_visual_perception(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process visual input for perception
        """
        image_data = parameters.get("image_data", "")
        image_format = parameters.get("format", "rgb8")
        camera_pose = parameters.get("camera_pose", {})

        # Simulate visual perception processing
        perception_results = {
            "detected_objects": [
                {"name": "cup", "confidence": 0.92, "position": [1.2, 0.5, 0.8]},
                {"name": "book", "confidence": 0.87, "position": [0.9, -0.3, 0.9]}
            ],
            "scene_description": "A room with a table containing a cup and a book",
            "camera_pose": camera_pose
        }

        return {
            "perception_results": perception_results,
            "image_format": image_format,
            "status": "processed"
        }

    async def _detect_objects(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect objects in the environment
        """
        image_data = parameters.get("image_data", "")
        object_classes = parameters.get("classes", ["cup", "book", "bottle"])

        # Simulate object detection
        detected_objects = []
        for i, obj_class in enumerate(object_classes):
            if i < 3:  # Simulate detecting first 3 objects
                detected_objects.append({
                    "class": obj_class,
                    "confidence": 0.8 + 0.1 * i,
                    "bbox": [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],  # [x, y, width, height]
                    "position_3d": [1.0 + 0.1 * i, 0.5 + 0.05 * i, 0.9]
                })

        return {
            "detected_objects": detected_objects,
            "total_detected": len(detected_objects),
            "status": "success"
        }

    async def _plan_manipulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan manipulation actions for objects
        """
        target_object = parameters.get("target_object", "")
        manipulation_type = parameters.get("manipulation_type", "grasp")
        environment_state = parameters.get("environment_state", {})

        # Plan manipulation sequence
        manipulation_plan = [
            {
                "action": ActionType.MOVE_TO,
                "parameters": {"target_object": target_object},
                "description": f"Approach the {target_object}"
            },
            {
                "action": ActionType.GRASP,
                "parameters": {"object": target_object},
                "description": f"Grasp the {target_object}"
            }
        ]

        if manipulation_type == "move":
            manipulation_plan.append({
                "action": ActionType.MOVE_TO,
                "parameters": {"target_location": parameters.get("target_location", "default")},
                "description": f"Move {target_object} to target location"
            })
            manipulation_plan.append({
                "action": ActionType.PLACE,
                "parameters": {"object": target_object},
                "description": f"Place the {target_object}"
            })

        return {
            "manipulation_plan": manipulation_plan,
            "target_object": target_object,
            "manipulation_type": manipulation_type,
            "status": "planned"
        }

    async def _process_general_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a general VLA task
        """
        return {
            "parameters": parameters,
            "status": "processed",
            "message": "General VLA task processed"
        }

    def get_current_plan(self) -> List[Dict[str, Any]]:
        """
        Get the current action plan
        """
        return self.current_plan

    def get_action_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of executed actions
        """
        return self.action_history