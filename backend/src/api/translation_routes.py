from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Dict
from ..database import get_db
from ..services.translation_service import TranslationService

router = APIRouter()
translation_service = TranslationService()


@router.get("/languages", response_model=Dict[str, str])
async def get_available_languages():
    """Get available target languages for translation."""
    return translation_service.get_available_languages()


@router.post("/translate", response_model=Dict[str, str])
async def translate_content(
    content: str,
    target_language: str = Query(..., description="Target language code (e.g., 'ur' for Urdu)"),
    source_language: str = Query("en", description="Source language code (default: 'en')"),
    preserve_formatting: bool = Query(True, description="Whether to preserve document formatting")
):
    """Translate content to the target language."""
    if preserve_formatting:
        translated_content = translation_service.translate_text_preserving_formatting(
            content, target_language
        )
    else:
        translated_content = translation_service.translate_content(
            content, target_language, source_language
        )

    return {
        "original_content": content,
        "translated_content": translated_content,
        "source_language": source_language,
        "target_language": target_language
    }


@router.post("/chapters/{chapter_id}/translate", response_model=Dict[str, str])
async def translate_chapter(
    chapter_id: str,
    target_language: str = Query(..., description="Target language code (e.g., 'ur' for Urdu)"),
    db: Session = Depends(get_db)
):
    """Translate a chapter to the target language."""
    translation_result = translation_service.translate_chapter(
        db, chapter_id, target_language
    )

    return {
        "id": translation_result["id"],
        "title": translation_result["title"],
        "content": translation_result["content"],
        "source_language": translation_result["source_language"],
        "target_language": translation_result["target_language"],
        "is_cached": translation_result["is_cached"]
    }