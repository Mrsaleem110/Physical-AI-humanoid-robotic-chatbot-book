#!/usr/bin/env python3
"""
Test script for Agentic Sphere Agent
"""
import asyncio
import json
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.agents.agentic_sphere_agent import AgenticSphereAgent
from src.agents.base_agent import AgentType


async def test_agentic_sphere_agent():
    """Test the Agentic Sphere agent functionality"""
    print("Testing Agentic Sphere Agent...")

    # Create an instance of the agent
    agent = AgenticSphereAgent()

    print(f"\nAgent created: {agent.name}")
    print(f"Agent type: {agent.agent_type}")
    print(f"Agent description: {agent.description}")
    print(f"CEO: {agent.ceo}")
    print(f"Vision: {agent.vision}")
    print(f"Mission: {agent.mission}")

    # Test agent info
    print("\nAgent Info:")
    agent_info = agent.get_agent_info()
    print(json.dumps(agent_info, indent=2))

    # Test different types of tasks
    test_tasks = [
        {
            "type": "info_request",
            "description": "Tell me about Agentic Sphere",
            "parameters": {}
        },
        {
            "type": "business_idea",
            "description": "Create an AI agent for customer service automation",
            "parameters": {
                "capabilities": ["conversation", "decision-making", "escalation"],
                "target": "reduce response time"
            }
        },
        {
            "type": "decision_making",
            "description": "Should we invest in AI automation?",
            "parameters": {
                "factors": ["cost", "efficiency", "ROI"],
                "options": ["yes", "no", "evaluate further"]
            }
        },
        {
            "type": "execution_planning",
            "description": "Plan the implementation of AI agents",
            "parameters": {
                "steps": ["research", "development", "testing", "deployment"],
                "timeline": "3 months"
            }
        }
    ]

    print("\nTesting various tasks:")
    for i, task in enumerate(test_tasks, 1):
        print(f"\n--- Task {i}: {task['type']} ---")
        result = await agent.execute(task)
        print(json.dumps(result, indent=2))

    # Test can_handle method
    print("\nTesting can_handle method:")
    test_queries = [
        {"type": "business_idea", "description": "I have a business idea for AI automation"},
        {"type": "agent_creation", "description": "Create an autonomous agent"},
        {"type": "decision_making", "description": "Help me make a business decision"},
        {"type": "general", "description": "What is the weather today?"},  # Should return False
        {"type": "info_request", "description": "Tell me about digital minds and autonomous AI"}
    ]

    for query in test_queries:
        can_handle = agent.can_handle(query)
        print(f"Query: '{query['description']}' -> Can handle: {can_handle}")

    print(f"\nAgentic Sphere Agent testing completed!")
    print(f"Business ideas processed: {len(agent.business_ideas)}")
    print(f"Agent tasks created: {len(agent.agent_tasks)}")
    print(f"Execution history entries: {len(agent.execution_history)}")


if __name__ == "__main__":
    asyncio.run(test_agentic_sphere_agent())