#!/usr/bin/env python3
"""
Test script to verify the fixed agentic sphere integration
"""
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.services.chat_service import ChatService

def test_query_detection():
    """Test the updated query detection method"""
    print("Testing updated query detection...")

    chat_service = ChatService()

    test_queries = [
        ("What is Agentic Sphere?", True),
        ("Tell me about Agentic Sphere", True),
        ("What is agentic?", True),
        ("Tell me about sphere", True),
        ("Who is Muhammad Saleem?", True),
        ("What is robotics?", False),
        ("Explain ROS2", False),
        ("Tell me about digital minds", True),
        ("How does autonomous agent work?", True),
    ]

    all_passed = True
    for query, expected in test_queries:
        result = chat_service._is_agentic_sphere_query(query)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{query}' -> {result} (expected {expected})")
        if result != expected:
            all_passed = False

    return all_passed

async def test_agent_processing():
    """Test the updated agent processing method"""
    print("\nTesting agent processing...")

    chat_service = ChatService()

    test_queries = [
        "What is Agentic Sphere?",
        "Tell me about Muhammad Saleem",
        "What does Agentic Sphere do?"
    ]

    for query in test_queries:
        print(f"  Processing: '{query}'")
        try:
            result = await chat_service._process_agentic_sphere_query(query)
            print(f"    Result length: {len(result)}")
            print(f"    Preview: {result[:100]}...")
            if "Agentic Sphere" in result:
                print(f"    ✓ Contains expected content")
            else:
                print(f"    ✗ Missing expected content")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing fixed agentic sphere integration...")

    detection_ok = test_query_detection()
    print()

    if detection_ok:
        print("Query detection passed, testing agent processing...")
        asyncio.run(test_agent_processing())
    else:
        print("Query detection failed, skipping agent processing test")