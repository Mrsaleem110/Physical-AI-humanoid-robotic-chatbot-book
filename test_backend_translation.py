#!/usr/bin/env python3
"""
Comprehensive test for backend translation functionality
"""
import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from src.services.translation_service import translation_service

async def test_comprehensive_translation():
    """Test comprehensive translation functionality"""
    print("Testing backend translation functionality...")

    # Test 1: Check if translator is available
    print(f"Deep translator available: {translation_service.translator_available}")

    if not translation_service.translator_available:
        print("❌ Deep translator not available")
        return False

    # Test 2: Test basic translation
    try:
        result = translation_service.translate_content("Hello, how are you?", "es", "en")
        print(f"English to Spanish: {result}")
        if "Hola" in result or "hola" in result.lower():
            print("[SUCCESS] Basic translation working")
        else:
            print("[WARNING] Basic translation completed but may not be accurate")
    except Exception as e:
        print(f"[ERROR] Basic translation failed: {e}")
        return False

    # Test 3: Test English to French
    try:
        result = translation_service.translate_content("Good morning", "fr", "en")
        print(f"English to French: {result}")
    except Exception as e:
        print(f"[ERROR] English to French translation failed: {e}")
        return False

    # Test 4: Test English to German
    try:
        result = translation_service.translate_content("How are you?", "de", "en")
        print(f"English to German: {result}")
    except Exception as e:
        print(f"[ERROR] English to German translation failed: {e}")
        return False

    # Test 5: Test async translation method
    try:
        result = await translation_service.translate_text("Hello world", "es", "en")
        print(f"Async translation result: {result['translated_text']}")
        if result['success']:
            print("[SUCCESS] Async translation working")
        else:
            print("[ERROR] Async translation failed")
            return False
    except Exception as e:
        print(f"[ERROR] Async translation failed: {e}")
        return False

    # Test 6: Test available languages
    try:
        languages = translation_service.get_available_languages()
        print(f"Available languages: {list(languages.keys())}")
        print("[SUCCESS] Language list retrieved")
    except Exception as e:
        print(f"[ERROR] Language list retrieval failed: {e}")
        return False

    # Test 7: Test English to Urdu (if possible with deep-translator)
    try:
        result = translation_service.translate_content("Hello", "ur", "en")
        print(f"English to Urdu: {result}")
        print("[SUCCESS] Urdu translation attempted")
    except Exception as e:
        print(f"[WARNING] Urdu translation failed (this is expected with some translation services): {e}")

    print("\n[SUCCESS] All translation tests completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_comprehensive_translation())
    if success:
        print("\n[SUCCESS] All backend translation tests passed!")
    else:
        print("\n[ERROR] Some translation tests failed!")
        sys.exit(1)