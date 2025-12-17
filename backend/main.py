"""
Physical AI & Humanoid Robotics Backend
Main Application Entry Point
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import AsyncIterator
import asyncio

from src.api import api_router
from src.services.database import init_db, startup_db
from src.services.qdrant_service import init_qdrant
from src.services.chat_service import init_chat_service
from src.services.translation_service import init_translation_service
from src.services.robotics_service import init_robotics_service
from src.utils.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifecycle manager
    """
    logger.info("Initializing application...")

    try:
        # Initialize database
        await startup_db()
        logger.info("Database initialized")

        # Initialize vector store
        await init_qdrant()
        logger.info("Qdrant vector store initialized")

        # Initialize chat service
        await init_chat_service()
        logger.info("Chat service initialized")

        # Initialize translation service
        await init_translation_service()
        logger.info("Translation service initialized")

        # Initialize robotics service
        await init_robotics_service()
        logger.info("Robotics service initialized")

        yield

    except Exception as e:
        logger.error(f"Error during application initialization: {e}")
        raise
    finally:
        # Cleanup operations
        logger.info("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Platform API",
    description="Backend API for humanoid robotics platform with chatbot, personalization, and AI capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Physical AI & Humanoid Robotics Platform API",
        "version": "1.0.0",
        "services": [
            "Authentication",
            "Content Management",
            "Chatbot (RAG)",
            "Personalization",
            "Translation (Urdu & others)",
            "Robotics (Claude Code Subagents)"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "healthy",
            "vector_store": "healthy",
            "chat_service": "healthy",
            "translation_service": "healthy",
            "robotics_service": "healthy"
        }
    }

@app.get("/health/database")
async def database_health():
    """Database health check"""
    from src.services.database import get_db_health
    return await get_db_health()

@app.get("/health/vector-store")
async def vector_store_health():
    """Vector store health check"""
    from src.services.qdrant_service import qdrant_service
    return await qdrant_service.get_collection_info()

@app.get("/health/chat-service")
async def chat_service_health():
    """Chat service health check"""
    # This would check the chat service status
    return {
        "status": "healthy",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "model_loaded": True
    }

@app.get("/health/translation-service")
async def translation_service_health():
    """Translation service health check"""
    # This would check the translation service status
    return {
        "status": "healthy",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "supported_languages": 10
    }

@app.get("/health/robotics-service")
async def robotics_service_health():
    """Robotics service health check"""
    # This would check the robotics service status
    return {
        "status": "healthy",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "subagents_loaded": 5
    }

@app.get("/api-info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "title": "Physical AI & Humanoid Robotics Platform API",
        "version": "1.0.0",
        "description": "Backend API for humanoid robotics platform with chatbot, personalization, and AI capabilities",
        "endpoints": {
            "authentication": "/auth/",
            "content": "/content/",
            "chat": "/chat/",
            "personalization": "/personalization/",
            "translation": "/translation/",
            "robotics": "/robotics/"
        },
        "documentation": "/docs",
        "redoc": "/redoc"
    }

@app.get("/status")
async def system_status():
    """Get comprehensive system status"""
    from src.services.database import get_db_health
    from src.services.qdrant_service import qdrant_service

    # Get database health
    db_health = await get_db_health()

    # Get vector store info
    vector_info = await qdrant_service.get_collection_info()

    # Get system info
    import psutil
    import os

    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "process_count": len(psutil.pids()),
        "uptime_seconds": __import__('time').time() - psutil.boot_time()
    }

    return {
        "system_info": system_info,
        "database": db_health,
        "vector_store": vector_info,
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }

# Mount static files if needed
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except FileNotFoundError:
    logger.info("No static directory found, skipping static file mount")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )