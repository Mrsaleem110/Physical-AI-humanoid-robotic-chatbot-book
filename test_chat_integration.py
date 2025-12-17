#!/usr/bin/env python3
"""
Test to check if chat service properly detects and handles agentic sphere queries
"""
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.services.chat_service import ChatService

async def test_chat_service_agentic_detection():
    """Test the chat service's agentic sphere detection and processing"""
    print("Testing Chat Service Agentic Sphere Integration...")

    try:
        # Create an instance of the chat service
        chat_service = ChatService()
        print("[OK] Chat service created")

        # Test the agentic sphere query detection method
        test_queries = [
            "What is Agentic Sphere?",
            "Tell me about Muhammad Saleem",
            "How does Agentic Sphere work?",
            "What is vision into intelligent action?",
            "Tell me about digital minds",
            "What is robotics?",  # This should NOT trigger agentic sphere
            "Explain ROS2 navigation",  # This should NOT trigger agentic sphere
        ]

        print("\nTesting query detection:")
        for query in test_queries:
            is_agentic = chat_service._is_agentic_sphere_query(query)
            print(f"  Query: '{query[:30]}...' -> Is Agentic: {is_agentic}")

        # Test the processing of an agentic sphere query directly
        print("\nTesting agentic sphere query processing:")
        agentic_query = "What is Agentic Sphere?"
        is_agentic = chat_service._is_agentic_sphere_query(agentic_query)
        print(f"  Is agentic query: {is_agentic}")

        if is_agentic:
            try:
                result = await chat_service._process_agentic_sphere_query(agentic_query)
                print(f"  [OK] Processing successful, result length: {len(result)}")
                print(f"  Result preview: {result[:100]}...")
            except Exception as e:
                print(f"  [ERROR] Processing failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [ERROR] Query not detected as agentic sphere")

    except Exception as e:
        print(f"[ERROR] Error testing chat service: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat_service_agentic_detection())