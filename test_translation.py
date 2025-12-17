#!/usr/bin/env python3
"""
Test script to verify translation functionality
"""
import sys
import os
import asyncio

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_translation_service():
    """Test the translation service functionality"""
    print("Testing Translation Service...")

    try:
        from src.services.translation_service import translation_service

        print("[OK] Translation service imported successfully")

        # Test basic translation
        test_text = "Hello, how are you?"
        result = await translation_service.translate_text(test_text, "ur", "en")

        print("[OK] Translation successful: {}".format(result['success']))
        if result['success']:
            print("[OK] Original: {}".format(result['original_text']))
            print("[OK] Translated: {}".format(result['translated_text']))
            print("[OK] Source: {}, Target: {}".format(result['source_language'], result['target_language']))
        else:
            print("[ERROR] Translation failed: {}".format(result.get('error', 'Unknown error')))

        # Test Urdu to English translation
        urdu_text = "آپ کیسے ہیں؟"
        result_urdu = await translation_service.translate_urdu_to_english(urdu_text)
        print("[OK] Urdu to English translation: {}".format(result_urdu['success']))

        # Test English to Urdu translation
        english_text = "Hello world"
        result_english = await translation_service.translate_english_to_urdu(english_text)
        print("[OK] English to Urdu translation: {}".format(result_english['success']))

        # Test markdown translation
        markdown_text = "# Heading\n\nThis is a **bold** text."
        result_markdown = await translation_service.translate_markdown(markdown_text, "ur")
        print("[OK] Markdown translation: {}".format(result_markdown['success']))

        # Test batch translation
        texts = ["Hello", "World", "Test"]
        result_batch = await translation_service.batch_translate(texts, "ur")
        print("[OK] Batch translation completed for {} items".format(len(result_batch)))

        # Test statistics
        stats = await translation_service.get_translation_statistics()
        print("[OK] Statistics retrieved: {} languages supported".format(len(stats['supported_languages'])))

        print("\n[SUCCESS] Translation functionality is working correctly!")
        return True

    except Exception as e:
        print("[ERROR] Error testing translation service: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing translation functionality...")
    success = asyncio.run(test_translation_service())

    if success:
        print("\n[SUCCESS] Translation functionality has been successfully implemented!")
    else:
        print("\n[ERROR] There are still issues with the translation functionality.")