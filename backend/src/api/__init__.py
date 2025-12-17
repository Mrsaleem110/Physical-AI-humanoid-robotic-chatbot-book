"""
API Routes for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter
from src.api import (
    auth_router,
    content_router,
    chat_router,
    personalization_router,
    translation_router,
    robotics_router
)

# Main API router
api_router = APIRouter()

# Include all API routes
api_router.include_router(auth_router, prefix="/auth", tags=["authentication"])
api_router.include_router(content_router, prefix="/content", tags=["content"])
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])
api_router.include_router(personalization_router, prefix="/personalization", tags=["personalization"])
api_router.include_router(translation_router, prefix="/translation", tags=["translation"])
api_router.include_router(robotics_router, prefix="/robotics", tags=["robotics"])

__all__ = [
    "api_router",
    "auth_router",
    "content_router",
    "chat_router",
    "personalization_router",
    "translation_router",
    "robotics_router"
]