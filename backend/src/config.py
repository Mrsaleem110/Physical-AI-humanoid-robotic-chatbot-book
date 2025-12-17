from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql://user:password@localhost:5432/humanoid_book"

    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # OpenAI settings
    openai_api_key: str = ""

    # Auth settings
    better_auth_secret: str = "your-secret-key-here"
    better_auth_url: str = "http://localhost:3000"

    # Application settings
    app_name: str = "Humanoid Robotics Book API"
    app_version: str = "1.0.0"
    debug: bool = False

    # CORS settings
    frontend_url: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()