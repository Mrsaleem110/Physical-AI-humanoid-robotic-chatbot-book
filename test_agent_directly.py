#!/usr/bin/env python3
"""
Simple test to check if Agentic Sphere agent works directly
"""
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.agents.agentic_sphere_agent import AgenticSphereAgent

async def test_agent_directly():
    """Test the Agentic Sphere agent directly"""
    print("Testing Agentic Sphere Agent directly...")

    try:
        # Create an instance of the agent
        agent = AgenticSphereAgent()
        print(f"[OK] Agent created successfully: {agent.name}")

        # Test the can_handle method
        test_task = {
            "type": "info_request",
            "description": "What is Agentic Sphere?",
            "parameters": {}
        }

        can_handle = agent.can_handle(test_task)
        print(f"[OK] Agent can handle task: {can_handle}")

        # Test the execute method
        result = await agent.execute(test_task)
        print(f"[OK] Agent execution successful: {result['success']}")

        if result['success']:
            print("[OK] Agent result: Success (not showing full result due to Unicode characters)")
            # Only print specific fields that don't contain Unicode
            result_data = result['result']
            if 'info' in result_data:
                info = result_data['info']
                print(f"[OK] Info found: {info.get('name', 'Unknown')}")
        else:
            print(f"[ERROR] Agent execution failed: {result.get('error', 'Unknown error')}")

        # Test the info request specifically
        info_task = {
            "type": "info_request",
            "description": "Tell me about Agentic Sphere",
            "parameters": {}
        }

        info_result = await agent.execute(info_task)
        print(f"[OK] Info request successful: {info_result['success']}")

        if info_result['success'] and 'info' in info_result['result']:
            info = info_result['result']['info']
            print(f"[OK] Retrieved info: {info['name']}, CEO: {info['ceo']}")
        else:
            print(f"[ERROR] Info request failed or no info returned")

    except Exception as e:
        print(f"[ERROR] Error testing agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent_directly())