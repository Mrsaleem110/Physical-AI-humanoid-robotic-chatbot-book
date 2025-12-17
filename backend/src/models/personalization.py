"""
Personalization Models for Physical AI & Humanoid Robotics Platform
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, relationship
from typing import Optional, Dict, Any
from pydantic import BaseModel

Base = declarative_base()

class UserPreference(Base):
    """
    User preferences for personalization
    """
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)

    # Learning preferences
    difficulty_level = Column(String, default="intermediate")  # beginner, intermediate, advanced
    learning_style = Column(String, default="visual")  # visual, auditory, kinesthetic
    content_format = Column(String, default="mixed")  # text, video, interactive, mixed
    update_frequency = Column(String, default="daily")  # daily, weekly, monthly

    # Notification preferences
    notification_preferences = Column(JSON)  # JSON field for notification settings

    # Adaptive learning preferences
    adaptive_preferences = Column(JSON)  # JSON field for adaptive settings

    # Personalization settings
    enable_personalization = Column(Boolean, default=True)
    enable_adaptive_content = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("User", back_populates="preferences")


class ContentPersonalization(Base):
    """
    Content personalization for specific content items
    """
    __tablename__ = "content_personalizations"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String, nullable=False)  # Could be chapter_id, module_id, etc.
    content_type = Column(String, nullable=False)  # chapter, module, exercise, etc.

    # User-specific personalization
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Personalized content adjustments
    difficulty_adjustment = Column(String, default="none")  # none, simplified, enriched
    learning_style_modifications = Column(JSON)  # Modifications based on learning style
    adaptive_content = Column(Text)  # Personalized content version
    content_modifications = Column(JSON)  # Specific modifications to content

    # Interaction data
    engagement_score = Column(Integer, default=0)  # Track user engagement
    time_spent = Column(Integer, default=0)  # Time spent on content in seconds
    completion_status = Column(String, default="not_started")  # not_started, in_progress, completed
    rating = Column(Integer)  # User rating of content (1-5)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="personalized_content")


# Add relationships to User model
from src.models.user import User
User.preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")
User.personalized_content = relationship("ContentPersonalization", back_populates="user", cascade="all, delete-orphan")


class UserPreferenceCreate(BaseModel):
    """Schema for creating user preferences"""
    difficulty_level: Optional[str] = "intermediate"
    learning_style: Optional[str] = "visual"
    content_format: Optional[str] = "mixed"
    update_frequency: Optional[str] = "daily"
    notification_preferences: Optional[Dict[str, Any]] = {}
    adaptive_preferences: Optional[Dict[str, Any]] = {}
    enable_personalization: Optional[bool] = True
    enable_adaptive_content: Optional[bool] = True


class UserPreferenceUpdate(BaseModel):
    """Schema for updating user preferences"""
    difficulty_level: Optional[str] = None
    learning_style: Optional[str] = None
    content_format: Optional[str] = None
    update_frequency: Optional[str] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    adaptive_preferences: Optional[Dict[str, Any]] = None
    enable_personalization: Optional[bool] = None
    enable_adaptive_content: Optional[bool] = None


class UserPreferenceResponse(BaseModel):
    """Schema for user preference response"""
    id: int
    user_id: int
    difficulty_level: str
    learning_style: str
    content_format: str
    update_frequency: str
    notification_preferences: Optional[Dict[str, Any]]
    adaptive_preferences: Optional[Dict[str, Any]]
    enable_personalization: bool
    enable_adaptive_content: bool
    created_at: str
    updated_at: Optional[str]

    class Config:
        from_attributes = True


class ContentPersonalizationCreate(BaseModel):
    """Schema for creating content personalization"""
    content_id: str
    content_type: str
    difficulty_adjustment: Optional[str] = "none"
    learning_style_modifications: Optional[Dict[str, Any]] = {}
    adaptive_content: Optional[str] = None
    content_modifications: Optional[Dict[str, Any]] = {}
    engagement_score: Optional[int] = 0
    time_spent: Optional[int] = 0
    completion_status: Optional[str] = "not_started"
    rating: Optional[int] = None


class ContentPersonalizationUpdate(BaseModel):
    """Schema for updating content personalization"""
    difficulty_adjustment: Optional[str] = None
    learning_style_modifications: Optional[Dict[str, Any]] = None
    adaptive_content: Optional[str] = None
    content_modifications: Optional[Dict[str, Any]] = None
    engagement_score: Optional[int] = None
    time_spent: Optional[int] = None
    completion_status: Optional[str] = None
    rating: Optional[int] = None


class ContentPersonalizationResponse(BaseModel):
    """Schema for content personalization response"""
    id: int
    content_id: str
    content_type: str
    user_id: int
    difficulty_adjustment: str
    learning_style_modifications: Optional[Dict[str, Any]]
    adaptive_content: Optional[str]
    content_modifications: Optional[Dict[str, Any]]
    engagement_score: int
    time_spent: int
    completion_status: str
    rating: Optional[int]
    created_at: str
    updated_at: Optional[str]

    class Config:
        from_attributes = True