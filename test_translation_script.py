#!/usr/bin/env python3
"""
Test script to verify translation functionality using deep-translator
"""
import os
import sys
from deep_translator import GoogleTranslator

def test_translation():
    """Test translation functionality"""
    try:
        # Test translation from English to Spanish (works with ASCII)
        translator = GoogleTranslator(source='en', target='es')
        result = translator.translate("Hello, how are you?")
        print(f"English to Spanish: {result}")

        # Test translation from English to French (works with ASCII)
        translator = GoogleTranslator(source='en', target='fr')
        result = translator.translate("Hello, how are you?")
        print(f"English to French: {result}")

        # Test translation from English to German (works with ASCII)
        translator = GoogleTranslator(source='en', target='de')
        result = translator.translate("Hello, how are you?")
        print(f"English to German: {result}")

        print("\nTranslation tests completed successfully!")
        return True

    except Exception as e:
        print(f"Error during translation: {e}")
        return False

if __name__ == "__main__":
    print("Testing translation functionality...")
    success = test_translation()
    if success:
        print("\n[SUCCESS] All translation tests passed!")
    else:
        print("\n[FAILED] Translation tests failed!")
        sys.exit(1)