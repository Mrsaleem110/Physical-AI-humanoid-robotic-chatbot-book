"""
Translation API Router for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime

from src.models.user import UserResponse
from src.models.translation import TranslationRequest, TranslationResponse
from src.services.database import get_db_session
from src.services.auth_service import get_current_user
from src.services.translation_service import translation_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
translation_router = APIRouter(
    prefix="/translation",
    tags=["translation"],
    responses={404: {"description": "Not found"}}
)

@translation_router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Translate text to specified language
    """
    try:
        # Perform translation
        result = await translation_service.translate_text(
            text=request.text,
            target_lang=request.target_language,
            source_lang=request.source_language
        )

        # Save to database for caching
        await translation_service.save_translation_to_db(
            db_session=db_session,
            original_text=result["original_text"],
            translated_text=result["translated_text"],
            source_lang=result["source_language"],
            target_lang=result["target_language"],
            content_type=request.content_type,
            user_id=current_user.id
        )

        return TranslationResponse(
            original_text=result["original_text"],
            translated_text=result["translated_text"],
            source_language=result["source_language"],
            target_language=result["target_language"],
            success=result["success"],
            error=result.get("error"),
            timestamp=result["timestamp"]
        )

    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation service error"
        )


@translation_router.post("/translate-content")
async def translate_content(
    content: str,
    target_lang: str = "ur",
    source_lang: str = "en",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Translate content (preserving structure where possible)
    """
    try:
        result = await translation_service.translate_content(
            content=content,
            target_lang=target_lang,
            source_lang=source_lang
        )

        return result

    except Exception as e:
        logger.error(f"Error translating content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation service error"
        )


@translation_router.post("/translate-markdown")
async def translate_markdown(
    markdown_content: str,
    target_lang: str = "ur",
    source_lang: str = "en",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Translate markdown content while preserving formatting
    """
    try:
        result = await translation_service.translate_markdown(
            markdown_content=markdown_content,
            target_lang=target_lang,
            source_lang=source_lang
        )

        return result

    except Exception as e:
        logger.error(f"Error translating markdown: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Markdown translation service error"
        )


@translation_router.post("/batch-translate")
async def batch_translate(
    texts: List[str],
    target_lang: str = "ur",
    source_lang: str = "en",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Translate multiple texts efficiently
    """
    try:
        results = await translation_service.batch_translate(
            texts=texts,
            target_lang=target_lang,
            source_lang=source_lang
        )

        return {
            "translations": results,
            "count": len(results),
            "target_language": target_lang,
            "source_language": source_lang,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in batch translation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch translation service error"
        )


@translation_router.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages
    """
    try:
        stats = await translation_service.get_translation_statistics()
        return {
            "supported_languages": stats["supported_languages"],
            "total_supported": len(stats["supported_languages"]),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving supported languages"
        )


@translation_router.get("/statistics")
async def get_translation_statistics():
    """
    Get translation service statistics
    """
    try:
        stats = await translation_service.get_translation_statistics()
        return stats

    except Exception as e:
        logger.error(f"Error getting translation statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving translation statistics"
        )


@translation_router.post("/translate-urdu-to-english")
async def translate_urdu_to_english(
    text: str,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Convenience endpoint to translate from Urdu to English
    """
    try:
        result = await translation_service.translate_urdu_to_english(text)
        return result

    except Exception as e:
        logger.error(f"Error translating Urdu to English: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Urdu to English translation error"
        )


@translation_router.post("/translate-english-to-urdu")
async def translate_english_to_urdu(
    text: str,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Convenience endpoint to translate from English to Urdu
    """
    try:
        result = await translation_service.translate_english_to_urdu(text)
        return result

    except Exception as e:
        logger.error(f"Error translating English to Urdu: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="English to Urdu translation error"
        )


@translation_router.get("/cache-status")
async def get_cache_status():
    """
    Get translation cache status
    """
    try:
        stats = await translation_service.get_translation_statistics()
        return {
            "cache_size": stats["cache_size"],
            "cache_entries": stats["cache_entries"],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting cache status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving cache status"
        )


@translation_router.post("/clear-cache")
async def clear_translation_cache(
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Clear translation cache (admin only)
    """
    try:
        # Check if user is admin
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required to clear cache"
            )

        await translation_service.clear_cache()

        return {
            "status": "success",
            "message": "Translation cache cleared",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing translation cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error clearing translation cache"
        )


@translation_router.post("/translate-document")
async def translate_document(
    document_content: str,
    target_lang: str = "ur",
    source_lang: str = "en",
    document_type: str = "text",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Translate entire documents with appropriate handling based on type
    """
    try:
        result = await translation_service.translate_document(
            document_content=document_content,
            target_lang=target_lang,
            source_lang=source_lang,
            document_type=document_type
        )

        return result

    except Exception as e:
        logger.error(f"Error translating document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document translation service error"
        )


@translation_router.get("/health")
async def translation_health():
    """
    Translation service health check
    """
    return {
        "status": "healthy",
        "service": "translation",
        "timestamp": datetime.utcnow().isoformat()
    }


__all__ = ["translation_router"]