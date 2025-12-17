from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from sqlalchemy import Column, String, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from .user import Base

Base = declarative_base()


class ChatSenderType(str, Enum):
    USER = "user"
    AI = "ai"


class ChatMessageType(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    SYSTEM = "system"


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Nullable for anonymous sessions
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    context_metadata = Column(JSON, nullable=True)  # Store chat context like "selected_text_only" mode

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id})>"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    sender_type = Column(String, nullable=False)  # "user" or "ai"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    message_type = Column(String, default="question", nullable=False)  # "question", "answer", "system"
    sources = Column(JSON, nullable=True)  # Store sources referenced in AI response

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, sender={self.sender_type})>"