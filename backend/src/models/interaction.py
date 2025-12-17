from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from sqlalchemy import Column, String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from .user import Base

Base = declarative_base()


class InteractionType(str, Enum):
    VIEW = "view"
    PERSONALIZE = "personalize"
    TRANSLATE = "translate"
    COMPLETE = "complete"
    CHAT = "chat"


class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    content_id = Column(String, nullable=False, index=True)  # References chapter/section ID
    content_type = Column(String, nullable=False)  # "chapter", "section", "exercise"
    interaction_type = Column(String, nullable=False)  # "view", "personalize", "translate", etc.
    interaction_data = Column(JSON, nullable=True)  # Additional data about the interaction
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<UserInteraction(id={self.id}, user_id={self.user_id}, type={self.interaction_type})>"