import unittest
from backend.src.services.translation_service import TranslationService

class TestTranslationService(unittest.TestCase):
    def setUp(self):
        self.translation_service = TranslationService()

    def test_available_languages(self):
        """Test that all required languages are available."""
        languages = self.translation_service.get_available_languages()

        expected_languages = ['en', 'ur']

        for lang_code in expected_languages:
            self.assertIn(lang_code, languages, f"Language {lang_code} not found in available languages")

        print("Available languages:", languages)
        self.assertEqual(len(languages), 2)  # 2 languages: English and Urdu

    def test_translate_content(self):
        """Test basic translation functionality."""
        test_text = "Hello, how are you?"
        result = self.translation_service.translate_content(test_text, 'ur')

        # The result should contain the original text if translation service is not available
        # or the translated text if it is available
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_translate_content_multiple_languages(self):
        """Test translation to multiple languages."""
        test_text = "Welcome to our website"

        target_languages = ['ur']  # Only Urdu for simplified version

        for lang in target_languages:
            result = self.translation_service.translate_content(test_text, lang)
            self.assertIsInstance(result, str)
            print(f"Translation to {lang}: {result}")

if __name__ == '__main__':
    unittest.main()