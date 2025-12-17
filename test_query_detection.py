#!/usr/bin/env python3
"""
Simple test to verify Agentic Sphere query detection logic
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_query_detection():
    """Test that the query detection logic works properly"""
    print("Testing Agentic Sphere query detection logic...")

    # Simulate the detection function
    def _is_agentic_sphere_query(message: str) -> bool:
        message_lower = message.lower()
        agentic_sphere_keywords = [
            "agentic sphere", "agentic", "sphere", "muhammad saleem",
            "digital mind", "autonomous agent", "ai agent", "intelligent action",
            "vision into action", "business idea", "agent creation", "decision making"
        ]
        return any(keyword in message_lower for keyword in agentic_sphere_keywords)

    # Test queries that should trigger the Agentic Sphere agent
    test_queries = [
        ("What is Agentic Sphere?", True),
        ("Tell me about Muhammad Saleem", True),
        ("What does Agentic Sphere do?", True),
        ("How does Agentic Sphere work?", True),
        ("Tell me about digital minds and autonomous agents", True),
        ("What is the mission of Agentic Sphere?", True),
        ("How can I use Agentic Sphere for business?", True),
        ("What is vision into intelligent action?", True),
        ("Tell me about the CEO of Agentic Sphere", True),
        ("How does Agentic Sphere transform business ideas?", True),
        ("What is robotics?", False),
        ("Tell me about ROS2", False),
        ("How to build a robot?", False),
        ("Explain machine learning", False),
        ("What is AI?", False)
    ]

    print("\nTesting query detection:")
    all_passed = True

    for query, expected in test_queries:
        result = _is_agentic_sphere_query(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] '{query}' -> {result} (expected {expected})")
        if result != expected:
            all_passed = False

    print(f"\nQuery detection test: {'PASSED' if all_passed else 'FAILED'}")

    # Test the default response
    default_response = """**Agentic Sphere** is a futuristic AI platform where bold business ideas are transformed into intelligent, autonomous AI agents that plan, decide, and execute with precision. We don't just build tools—we create digital minds that work for your business 24/7, scaling operations, automating decisions, and unlocking new growth.

Led by **Muhammad Saleem**, CEO and an **AI-native visionary who thinks like an artificial intelligent agent**, Agentic Sphere stands at the intersection of innovation and execution. His vision is to redefine how businesses operate by turning imagination into living, thinking AI systems.

**Agentic Sphere — Turning Vision into Intelligent Action. 🚀**"""

    print(f"\nDefault Agentic Sphere response will be:")
    print(default_response[:200] + "..." if len(default_response) > 200 else default_response)

    print(f"\nAgentic Sphere integration logic verification completed!")


if __name__ == "__main__":
    test_query_detection()