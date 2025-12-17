"""
Database Models for Physical AI & Humanoid Robotics Platform
"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

# Import all models to ensure they are registered with SQLAlchemy
from .user import User
from .chapter import Chapter, Module
from .chat_session import ChatSession, ChatMessage
from .personalization import UserPreference, ContentPersonalization
from .translation import Translation
from .robotics import RoboticsTask, RobotState

__all__ = [
    "Base",
    "User",
    "Chapter",
    "Module",
    "ChatSession",
    "ChatMessage",
    "UserPreference",
    "ContentPersonalization",
    "Translation",
    "RoboticsTask",
    "RobotState"
]