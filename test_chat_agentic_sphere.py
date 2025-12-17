#!/usr/bin/env python3
"""
Test script to verify Agentic Sphere integration with chatbot
"""
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.services.chat_service import chat_service
from src.services.robotics_service import robotics_service


async def test_agentic_sphere_chat_integration():
    """Test that the chat service properly routes Agentic Sphere queries to the agent"""
    print("Testing Agentic Sphere integration with chatbot...")

    # Initialize the services
    await chat_service.initialize()

    # Test queries that should trigger the Agentic Sphere agent
    test_queries = [
        "What is Agentic Sphere?",
        "Tell me about Muhammad Saleem",
        "What does Agentic Sphere do?",
        "How does Agentic Sphere work?",
        "Tell me about digital minds and autonomous agents",
        "What is the mission of Agentic Sphere?",
        "How can I use Agentic Sphere for business?",
        "What is vision into intelligent action?",
        "Tell me about the CEO of Agentic Sphere",
        "How does Agentic Sphere transform business ideas?"
    ]

    print("\nTesting Agentic Sphere queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")

        # Check if the query is detected as Agentic Sphere related
        is_agentic_query = chat_service._is_agentic_sphere_query(query)
        print(f"Is Agentic Sphere query: {is_agentic_query}")

        # Process the query through the chat service
        response = await chat_service.process_chat_message(
            user_id=1,
            message=query,
            session_id=f"test_session_{i}"
        )

        print(f"Response: {response['response'][:200]}...")
        if len(response['response']) > 200:
            print("... (response truncated for display)")

    # Test a non-Agentic Sphere query to ensure normal functionality still works
    print(f"\n--- Testing non-Agentic Sphere query ---")
    normal_query = "What is robotics?"
    is_agentic_query = chat_service._is_agentic_sphere_query(normal_query)
    print(f"Is Agentic Sphere query: {is_agentic_query}")

    response = await chat_service.process_chat_message(
        user_id=1,
        message=normal_query,
        session_id="test_session_normal"
    )

    print(f"Response: {response['response'][:200]}...")
    if len(response['response']) > 200:
        print("... (response truncated for display)")

    print(f"\nAgentic Sphere chat integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_agentic_sphere_chat_integration())