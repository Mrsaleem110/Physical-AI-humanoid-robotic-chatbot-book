"""
Configuration Settings for Physical AI & Humanoid Robotics Platform
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # Application
    APP_NAME: str = "Physical AI & Humanoid Robotics Platform"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/humanoid_db")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_POOL_MAX_OVERFLOW: int = 30

    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]  # Change in production

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4-turbo-preview"

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME: str = "humanoid_documents"

    # Vector embeddings
    EMBEDDING_DIMENSION: int = 1536  # For OpenAI embeddings
    SIMILARITY_THRESHOLD: float = 0.7

    # Translation
    TRANSLATE_TO_URDU: bool = True
    TRANSLATION_SERVICE: str = "deep_translator"  # google, deep_translator, etc.

    # Robotics
    ROBOTICS_API_BASE: str = "http://localhost:8080"
    ROBOTICS_TIMEOUT: int = 30

    # Personalization
    PERSONALIZATION_ENABLED: bool = True
    CONTENT_ADAPTATION_LEVELS: List[str] = ["beginner", "intermediate", "advanced"]

    # File uploads
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["txt", "pdf", "doc", "docx", "md"]

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Cache
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """
    Validate that required settings are present
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required")

    if not settings.SECRET_KEY or settings.SECRET_KEY == "dev-secret-key-change-in-production":
        print("WARNING: Using default secret key. Change SECRET_KEY in production!")


# Validate settings on import
validate_settings()

__all__ = ["settings", "Settings", "validate_settings"]