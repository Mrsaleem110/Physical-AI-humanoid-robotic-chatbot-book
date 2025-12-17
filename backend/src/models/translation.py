from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel, Field
from .user import Base

Base = declarative_base()


class Translation(Base):
    __tablename__ = "translations"

    id = Column(String, primary_key=True, index=True)
    content_id = Column(String, nullable=False, index=True)  # References chapter/section ID
    content_type = Column(String, nullable=False)  # "chapter", "section", "paragraph"
    source_language = Column(String, default="en", nullable=False)
    target_language = Column(String, nullable=False)  # e.g., "ur" for Urdu
    original_content = Column(Text, nullable=False)
    translated_content = Column(Text, nullable=False)
    translation_quality_score = Column(Float, nullable=True)  # 0-1 scale
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self):
        return f"<Translation(id={self.id}, source={self.source_language}, target={self.target_language})>"


# Pydantic models for API requests and responses
class TranslationRequest(BaseModel):
    text: str
    target_language: str = Field(default="ur", description="Target language code (e.g., 'ur', 'hi', 'fr', 'de', 'zh', 'ja')")
    source_language: str = Field(default="en", description="Source language code")
    content_type: str = Field(default="text", description="Type of content being translated")


class TranslationResponse(BaseModel):
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    success: bool
    error: Optional[str] = None
    timestamp: str