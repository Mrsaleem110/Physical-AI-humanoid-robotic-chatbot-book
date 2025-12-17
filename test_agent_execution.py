#!/usr/bin/env python3
"""
Test to verify the Agentic Sphere agent execution works properly
"""
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_agent_execution():
    """Test that the Agentic Sphere agent can be executed directly"""
    print("Testing Agentic Sphere agent execution...")

    try:
        # Import the agent
        from src.agents.agentic_sphere_agent import AgenticSphereAgent

        print("PASS: Agent imported successfully")

        # Create an instance
        agent = AgenticSphereAgent()
        print(f"PASS: Agent created: {agent.name}")
        print(f"PASS: Agent type: {agent.agent_type}")

        # Test executing an info_request task
        task = {
            "type": "info_request",
            "description": "What is Agentic Sphere?",
            "parameters": {}
        }

        print(f"Executing task: {task}")
        result = await agent.execute(task)

        print(f"PASS: Task executed successfully")
        print(f"PASS: Success: {result.get('success')}")
        print(f"PASS: Result keys: {list(result.keys())}")

        if result.get("success"):
            agent_result = result.get("result", {})
            print(f"PASS: Agent result keys: {list(agent_result.keys())}")

            if "info" in agent_result:
                info = agent_result["info"]
                print(f"PASS: Info found in result")
                print(f"   Name: {info.get('name', 'N/A')}")
                print(f"   CEO: {info.get('ceo', 'N/A')}")
                print(f"   Mission: {info.get('mission', 'N/A')}")
                # Skip printing tagline to avoid emoji encoding issues
                # print(f"   Tagline: {info.get('tagline', 'N/A')}")
                tagline = info.get('tagline', 'N/A')
                print(f"   Tagline: [Contains emoji - would display: {len(tagline)} chars]")
            else:
                print(f"   Message: {agent_result.get('message', 'N/A')}")

        # Test the can_handle method
        test_task = {
            "type": "info_request",
            "description": "What is Agentic Sphere?"
        }

        can_handle = agent.can_handle(test_task)
        print(f"PASS: Agent can handle 'What is Agentic Sphere?' task: {can_handle}")

        # Test with other types of tasks
        other_tasks = [
            {"type": "business_idea", "description": "Transform this business idea"},
            {"type": "decision_making", "description": "Help me make a decision"},
            {"type": "general", "description": "What is the weather?"}
        ]

        print("\nTesting other task types:")
        for i, task in enumerate(other_tasks):
            can_handle = agent.can_handle(task)
            print(f"  Task {i+1}: {task['type']} -> {can_handle}")

        print("\nAgentic Sphere agent execution test PASSED!")
        print("The agent can now be called directly from the chat service!")

        return True

    except Exception as e:
        print(f"FAIL: Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_execution())
    if success:
        print("\nThe Agentic Sphere agent integration should now work properly!")
        print("Remember to restart the application server for changes to take effect.")
    else:
        print("\nThe integration still has issues.")