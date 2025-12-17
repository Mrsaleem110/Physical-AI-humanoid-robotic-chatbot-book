#!/usr/bin/env python3
"""
Simple test to verify the agentic sphere functionality works independently
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_agentic_sphere_directly():
    """Test the agentic sphere functionality directly without full service initialization"""
    print("Testing Agentic Sphere functionality directly...")

    try:
        # Test importing the agentic sphere agent directly
        from src.agents.agentic_sphere_agent import AgenticSphereAgent
        print("[OK] AgenticSphereAgent imported successfully")

        # Test creating an instance
        agent = AgenticSphereAgent()
        print(f"[OK] AgenticSphereAgent created: {agent.name}")

        # Test the can_handle method
        test_task = {
            "type": "info_request",
            "description": "What is Agentic Sphere?",
            "parameters": {}
        }
        can_handle = agent.can_handle(test_task)
        print(f"[OK] Agent can handle agentic sphere query: {can_handle}")

        # Test the execute method (async)
        import asyncio

        async def test_execute():
            result = await agent.execute(test_task)
            print(f"[OK] Agent execution successful: {result['success']}")
            if result['success']:
                print("[OK] Agentic sphere agent is working correctly")
                return True
            else:
                print(f"[ERROR] Agent execution failed: {result.get('error', 'Unknown error')}")
                return False

        success = asyncio.run(test_execute())
        return success

    except Exception as e:
        print(f"[ERROR] Error testing agentic sphere: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_service_agentic_detection():
    """Test just the agentic sphere detection functionality"""
    print("\nTesting chat service agentic sphere detection...")

    try:
        # Temporarily set a mock settings to avoid config issues
        import os
        os.environ.setdefault('OPENAI_API_KEY', 'mock_key')
        os.environ.setdefault('ALLOWED_ORIGINS', '["*"]')  # Set as JSON array

        from src.services.chat_service import ChatService
        print("[OK] ChatService imported successfully")

        # Create instance
        chat_service = ChatService()
        print("[OK] ChatService instance created")

        # Test agentic sphere query detection
        test_queries = [
            ("What is Agentic Sphere?", True),
            ("Tell me about Muhammad Saleem", True),
            ("What is robotics?", False),
            ("Explain agentic systems", True)
        ]

        all_passed = True
        for query, expected in test_queries:
            result = chat_service._is_agentic_sphere_query(query)
            status = "[OK]" if result == expected else "[ERROR]"
            print(f"  {status} Query: '{query}' -> {result} (expected {expected})")
            if result != expected:
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"[ERROR] Error testing chat service: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing agentic sphere functionality after dependency resolution...")

    agent_ok = test_agentic_sphere_directly()
    detection_ok = test_chat_service_agentic_detection()

    if agent_ok and detection_ok:
        print("\n[SUCCESS] Agentic sphere functionality is working correctly!")
        print("The integration should now work when the backend starts up properly.")
    else:
        print("\n[ERROR] There are still issues with the agentic sphere functionality.")