from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentType
from .research_agent import ResearchAgent
from .ros2_agent import ROS2Agent
from .simulation_agent import SimulationAgent
from .vla_agent import VLAActionPlanningAgent
from .retrieval_agent import RetrievalAgent
from .personalization_agent import PersonalizationAgent
from .agentic_sphere_agent import AgenticSphereAgent
import asyncio


class AgentCoordinator:
    """
    Coordinates communication between different Claude Code Subagents
    """

    def __init__(self):
        self.agents: Dict[AgentType, BaseAgent] = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all subagents"""
        self.agents[AgentType.RESEARCH] = ResearchAgent()
        self.agents[AgentType.ROS2] = ROS2Agent()
        self.agents[AgentType.SIMULATION] = SimulationAgent()
        self.agents[AgentType.VLA] = VLAActionPlanningAgent()
        self.agents[AgentType.RETRIEVAL] = RetrievalAgent()
        self.agents[AgentType.PERSONALIZATION] = PersonalizationAgent()
        self.agents[AgentType.AGENTIC_SPHERE] = AgenticSphereAgent()

    async def route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a task to the most appropriate agent
        """
        # Determine the best agent for the task
        best_agent = self._find_best_agent(task)

        if best_agent:
            result = await best_agent.execute(task)
            return {
                "agent_type": best_agent.agent_type.value,
                "result": result,
                "status": "success"
            }
        else:
            return {
                "error": "No suitable agent found for the task",
                "status": "error"
            }

    def _find_best_agent(self, task: Dict[str, Any]) -> Optional[BaseAgent]:
        """
        Find the most appropriate agent for a given task
        """
        task_type = task.get("type", "")
        task_description = task.get("description", "")
        required_skills = task.get("required_skills", [])

        # Score each agent based on its ability to handle the task
        agent_scores = []
        for agent_type, agent in self.agents.items():
            score = self._calculate_agent_score(agent, task, task_type, task_description, required_skills)
            agent_scores.append((agent, score))

        # Sort by score and return the best agent if score is above threshold
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        best_agent, best_score = agent_scores[0]

        # Only return agent if score is above threshold (0.5)
        return best_agent if best_score > 0.5 else None

    def _calculate_agent_score(self, agent: BaseAgent, task: Dict[str, Any],
                              task_type: str, task_description: str, required_skills: List[str]) -> float:
        """
        Calculate how well an agent can handle a task
        """
        score = 0.0

        # Check if agent can handle the task type
        if agent.can_handle(task):
            score += 0.4  # Base score for capability

        # Check if task type matches agent specialization
        if task_type and agent.agent_type.value in task_type.lower():
            score += 0.3

        # Check if task description mentions agent's domain
        description_lower = task_description.lower()
        agent_domain_keywords = self._get_agent_domain_keywords(agent.agent_type)
        if any(keyword in description_lower for keyword in agent_domain_keywords):
            score += 0.2

        # Check if agent has required skills
        agent_skills = [skill.value for skill in agent.get_skills()]
        required_skills_met = sum(1 for skill in required_skills if skill in agent_skills)
        if required_skills:
            score += (required_skills_met / len(required_skills)) * 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _get_agent_domain_keywords(self, agent_type: AgentType) -> List[str]:
        """
        Get domain-specific keywords for each agent type
        """
        keywords_map = {
            AgentType.RESEARCH: ["research", "study", "analyze", "data", "information", "literature", "search"],
            AgentType.ROS2: ["robot", "ros", "node", "topic", "service", "control", "navigation", "movement"],
            AgentType.SIMULATION: ["simulation", "physics", "environment", "sensor", "gazebo", "unity", "isaac", "visualize"],
            AgentType.VLA: ["vision", "language", "action", "plan", "manipulation", "grasp", "place", "move"],
            AgentType.RETRIEVAL: ["retrieve", "search", "find", "query", "knowledge", "context", "information"],
            AgentType.PERSONALIZATION: ["personalize", "adapt", "customize", "profile", "recommend", "learning", "path"],
            AgentType.AGENTIC_SPHERE: ["business", "idea", "agent", "plan", "execute", "decision", "scale", "growth", "automation", "ai", "intelligent", "digital mind", "vision", "execution", "autonomous", "agentic", "sphere"]
        }
        return keywords_map.get(agent_type, [])

    async def execute_multi_agent_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task that requires coordination between multiple agents
        """
        subtasks = task.get("subtasks", [])
        dependencies = task.get("dependencies", {})
        results = {}

        # Execute subtasks in dependency order
        completed = set()
        max_iterations = len(subtasks) * 2  # Prevent infinite loops
        iteration = 0

        while len(completed) < len(subtasks) and iteration < max_iterations:
            iteration += 1
            executed_in_iteration = False

            for i, subtask in enumerate(subtasks):
                if i in completed:
                    continue

                # Check if all dependencies are completed
                deps = dependencies.get(i, [])
                if all(dep in completed for dep in deps):
                    # Execute subtask
                    result = await self.route_task(subtask)
                    results[i] = result
                    completed.add(i)
                    executed_in_iteration = True

            if not executed_in_iteration:
                # No progress made, likely a circular dependency
                break

        return {
            "results": results,
            "completed_count": len(completed),
            "total_count": len(subtasks),
            "status": "completed" if len(completed) == len(subtasks) else "partial"
        }

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get information about all agents
        """
        return {
            agent_type.value: agent.get_agent_info()
            for agent_type, agent in self.agents.items()
        }

    async def broadcast_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast a message to all agents
        """
        results = {}
        for agent_type, agent in self.agents.items():
            try:
                # Try to execute with the agent, but ignore if it can't handle the message
                if agent.can_handle(message):
                    result = await agent.execute(message)
                    results[agent_type.value] = result
            except Exception as e:
                results[agent_type.value] = {
                    "status": "error",
                    "error": str(e)
                }

        return {
            "results": results,
            "broadcasted_message": message
        }

    async def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents
        """
        status = {}
        for agent_type, agent in self.agents.items():
            # In a real implementation, you'd check actual agent status
            status[agent_type.value] = {
                "status": "ready",
                "tasks_handled": 0,  # Would track actual metrics
                "errors": 0,  # Would track actual metrics
                "memory_usage": "low"  # Would check actual memory
            }

        return {
            "status": status,
            "total_agents": len(self.agents)
        }