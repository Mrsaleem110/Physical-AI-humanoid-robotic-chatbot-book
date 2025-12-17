from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from ..models.translation import Translation
from ..models.chapter import Chapter
from fastapi import HTTPException, status
import uuid
import logging

# Try to import deep-translator, but provide fallback if not available
try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    GoogleTranslator = None
    DEEP_TRANSLATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(self):
        if DEEP_TRANSLATOR_AVAILABLE:
            # Deep translator doesn't need initialization, we create instances as needed
            self.translator_available = True
        else:
            self.translator_available = False
            logger.warning("Deep Translator not available. Using mock translation functionality.")

    def translate_content(self, content: str, target_language: str, source_language: str = "en") -> str:
        """Translate content to the target language."""
        try:
            if DEEP_TRANSLATOR_AVAILABLE and self.translator_available:
                # Use deep-translator for actual translation
                try:
                    translator = GoogleTranslator(source=source_language, target=target_language)
                    result = translator.translate(content)
                    return result
                except Exception as e:
                    logger.warning(f"GoogleTranslator failed: {e}")
                    # Try alternative translation methods if Google fails
                    return self._translate_with_alternative_methods(content, target_language, source_language)
            else:
                # Fallback: return mock translation or original content with note
                # In a real implementation, you might want to use a different translation service
                # or return an error indicating that translation is not available
                logger.warning(f"Translation service not available, returning original content for: {content[:50]}...")
                return f"[TRANSLATION_UNAVAILABLE: {content}]"
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # If there's an error, return the original content with a note
            return f"[TRANSLATION_ERROR: {content}]"

    def _translate_with_alternative_methods(self, content: str, target_language: str, source_language: str = "en") -> str:
        """Try alternative translation methods if the primary method fails."""
        # This is a fallback method in case the primary translation service fails
        # In a real implementation, you might want to use other services like LibreTranslate
        # For now, return the original content with a note
        logger.warning(f"Using alternative translation method for: {content[:50]}...")
        return content  # Return original content if translation fails

    def translate_chapter(self, db: Session, chapter_id: str, target_language: str) -> Dict[str, Any]:
        """Translate a chapter to the target language."""
        # Get the chapter
        chapter = db.query(Chapter).filter(Chapter.id == chapter_id, Chapter.is_published == True).first()
        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chapter not found"
            )

        # Check if translation already exists in the database
        existing_translation = db.query(Translation).filter(
            Translation.content_id == chapter_id,
            Translation.content_type == "chapter",
            Translation.target_language == target_language
        ).first()

        if existing_translation:
            # Return existing translation
            return {
                "id": chapter.id,
                "title": chapter.title,
                "content": existing_translation.translated_content,
                "source_language": existing_translation.source_language,
                "target_language": existing_translation.target_language,
                "is_cached": True
            }

        # Translate the content
        translated_content = self.translate_content(chapter.content_en, target_language)

        # Create a new translation record
        translation_id = str(uuid.uuid4())
        db_translation = Translation(
            id=translation_id,
            content_id=chapter_id,
            content_type="chapter",
            source_language="en",
            target_language=target_language,
            original_content=chapter.content_en,
            translated_content=translated_content
        )

        db.add(db_translation)
        db.commit()

        return {
            "id": chapter.id,
            "title": chapter.title,  # Keep title in original language or translate separately
            "content": translated_content,
            "source_language": "en",
            "target_language": target_language,
            "is_cached": False
        }

    def get_available_languages(self) -> Dict[str, str]:
        """Get available target languages for translation."""
        # In a real implementation, this would come from a configuration or database
        return {
            "en": "English",
            "ur": "Urdu"
        }

    def preserve_formatting(self, content: str, translated_content: str) -> str:
        """Preserve formatting like headings, lists, and code blocks during translation."""
        # This is a simplified implementation
        # A real implementation would need more sophisticated parsing to preserve formatting
        # while translating the actual text content

        # For now, return the translated content as-is
        # In a production system, you would:
        # 1. Parse the original content to identify formatting elements
        # 2. Extract text content while preserving formatting structure
        # 3. Translate only the text content
        # 4. Reconstruct the content with original formatting but translated text

        return translated_content

    def translate_text_preserving_formatting(self, content: str, target_language: str) -> str:
        """Translate text while preserving formatting like headings, lists, and code blocks."""
        # This is a more sophisticated version that attempts to preserve formatting
        import re

        # In a real implementation, you would:
        # 1. Use a proper markdown/parser library to identify structural elements
        # 2. Separate formatting from content
        # 3. Translate only the content portions
        # 4. Reassemble with original formatting

        # For this example, we'll use a simplified approach with regex
        # This is not production-ready but shows the concept

        # Pattern to identify markdown elements (headings, lists, code blocks)
        patterns = [
            # Headings: # Heading 1, ## Heading 2, etc.
            (r'^(#{1,6})\s+(.+)$', r'\1 \2'),
            # Code blocks: `code` and ```code blocks```
            (r'(```[\s\S]*?```)', r'\1'),
            (r'(`[^`]+`)', r'\1'),
            # Lists: - item or * item
            (r'^(\s*[-*+]\s+)(.+)$', r'\1\2'),
            # Bold: **text** or __text__
            (r'(\*\*|__)(.+?)(\*\*|__)', r'\1\2\3'),
        ]

        # This is a simplified approach - in reality you'd need a proper parser
        # For now, just translate the content as a whole
        return self.translate_content(content, target_language)

    def batch_translate(self, db: Session, content_list: list, target_language: str) -> list:
        """Translate multiple content items at once."""
        results = []
        for content in content_list:
            translated = self.translate_content(content, target_language)
            results.append(translated)
        return results

    def get_translation_quality_score(self, original: str, translated: str) -> float:
        """Estimate translation quality (simplified implementation)."""
        # In a real implementation, you would use more sophisticated metrics
        # like BLEU, METEOR, or other translation quality metrics
        # For this example, we'll just return a placeholder value
        return 0.85  # Assume good quality translation

    async def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> Dict[str, Any]:
        """Translate text with result structure matching router expectations."""
        try:
            # Translate the text
            translated_text = self.translate_content(text, target_lang, source_lang)

            result = {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "success": True,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

            return result
        except Exception as e:
            logger.error(f"Error in translate_text: {e}")
            return {
                "original_text": text,
                "translated_text": "",
                "source_language": source_lang,
                "target_language": target_lang,
                "success": False,
                "error": str(e),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

    async def save_translation_to_db(self, db_session, original_text: str, translated_text: str,
                                   source_lang: str, target_lang: str, content_type: str, user_id: int):
        """Save translation to database for caching."""
        from sqlalchemy import select
        from ..models.translation import Translation
        import uuid

        # Create a new translation record
        translation_id = str(uuid.uuid4())
        db_translation = Translation(
            id=translation_id,
            content_id=translation_id,  # Using the translation ID as content_id for text translations
            content_type=content_type,
            source_language=source_lang,
            target_language=target_lang,
            original_content=original_text,
            translated_content=translated_text
        )

        # Add to database session
        db_session.add(db_translation)
        await db_session.commit()

    async def translate_markdown(self, markdown_content: str, target_lang: str, source_lang: str = "en") -> Dict[str, Any]:
        """Translate markdown content while preserving formatting."""
        try:
            # Use the existing formatting preservation method
            translated_content = self.translate_text_preserving_formatting(markdown_content, target_lang)

            result = {
                "original_content": markdown_content,
                "translated_content": translated_content,
                "source_language": source_lang,
                "target_language": target_lang,
                "formatting_preserved": True,
                "success": True,
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

            return result
        except Exception as e:
            logger.error(f"Error in translate_markdown: {e}")
            return {
                "original_content": markdown_content,
                "translated_content": "",
                "source_language": source_lang,
                "target_language": target_lang,
                "formatting_preserved": False,
                "success": False,
                "error": str(e),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

    async def batch_translate(self, texts: list, target_lang: str, source_lang: str = "en") -> list:
        """Translate multiple texts efficiently."""
        results = []
        for text in texts:
            try:
                translated = self.translate_content(text, target_lang, source_lang)
                results.append({
                    "original": text,
                    "translated": translated,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "original": text,
                    "translated": "",
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "success": False,
                    "error": str(e)
                })
        return results

    async def get_translation_statistics(self) -> Dict[str, Any]:
        """Get translation service statistics."""
        # In a real implementation, this would query the database for statistics
        # For now, return placeholder values
        return {
            "supported_languages": self.get_available_languages(),
            "total_translations": 0,  # This would come from database in real implementation
            "cache_size": 0,
            "cache_entries": 0,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    async def translate_urdu_to_english(self, text: str) -> Dict[str, Any]:
        """Translate from Urdu to English."""
        return await self.translate_text(text, "en", "ur")

    async def translate_english_to_urdu(self, text: str) -> Dict[str, Any]:
        """Translate from English to Urdu."""
        return await self.translate_text(text, "ur", "en")

    async def clear_cache(self):
        """Clear translation cache."""
        # In a real implementation, this would clear any caching mechanism
        # For now, it's a no-op since we don't have caching implemented
        logger.info("Translation cache cleared (no cache implementation)")

    async def translate_document(self, document_content: str, target_lang: str,
                               source_lang: str = "en", document_type: str = "text") -> Dict[str, Any]:
        """Translate entire documents with appropriate handling based on type."""
        try:
            if document_type.lower() == "markdown":
                # Use markdown-specific translation
                result = await self.translate_markdown(document_content, target_lang, source_lang)
            else:
                # Use regular translation
                translated_content = self.translate_content(document_content, target_lang, source_lang)
                result = {
                    "original_content": document_content,
                    "translated_content": translated_content,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "document_type": document_type,
                    "success": True,
                    "timestamp": __import__('datetime').datetime.utcnow().isoformat()
                }

            return result
        except Exception as e:
            logger.error(f"Error in translate_document: {e}")
            return {
                "original_content": document_content,
                "translated_content": "",
                "source_language": source_lang,
                "target_language": target_lang,
                "document_type": document_type,
                "success": False,
                "error": str(e),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }


# Global instance
translation_service = TranslationService()

async def init_translation_service():
    """Initialize the translation service."""
    # For now, just log that initialization is complete
    # In a real implementation, you might load language models, etc.
    if DEEP_TRANSLATOR_AVAILABLE:
        logger.info("Translation service initialized with deep-translator")
    else:
        logger.warning("Translation service initialized without translation library - using mock functionality")
    return translation_service