from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentType(str, Enum):
    RESEARCH = "research"
    ROS2 = "ros2"
    SIMULATION = "simulation"
    VLA = "vla"
    RETRIEVAL = "retrieval"
    PERSONALIZATION = "personalization"
    AGENTIC_SPHERE = "agentic_sphere"


class AgentSkill(str, Enum):
    """Standardized skills that agents can have"""
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    PLANNING = "planning"
    EXECUTION = "execution"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    ADAPTATION = "adaptation"
    REASONING = "reasoning"


class BaseAgent(ABC):
    """
    Base class for all Claude Code Subagents
    Defines the standard interface for agent communication and execution
    """

    def __init__(self, agent_type: AgentType, name: str, description: str):
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.skills: List[AgentSkill] = []
        self.memory = {}
        self.config = {}

    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task and return the result
        """
        pass

    @abstractmethod
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """
        Determine if this agent can handle the given task
        """
        pass

    def add_skill(self, skill: AgentSkill):
        """Add a skill to the agent"""
        if skill not in self.skills:
            self.skills.append(skill)

    def get_skills(self) -> List[AgentSkill]:
        """Get the agent's skills"""
        return self.skills.copy()

    def update_memory(self, key: str, value: Any):
        """Update the agent's memory"""
        self.memory[key] = value

    def get_from_memory(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent's memory"""
        return self.memory.get(key, default)

    def update_config(self, config: Dict[str, Any]):
        """Update the agent's configuration"""
        self.config.update(config)

    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration"""
        return self.config.copy()

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent"""
        return {
            "type": self.agent_type,
            "name": self.name,
            "description": self.description,
            "skills": [skill.value for skill in self.skills],
            "memory_keys": list(self.memory.keys()),
            "config_keys": list(self.config.keys())
        }