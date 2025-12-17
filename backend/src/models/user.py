from datetime import datetime
from typing import Optional
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLEnum, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class BackgroundLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    NONE = "none"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    software_background = Column(SQLEnum(BackgroundLevel), nullable=False, default=BackgroundLevel.NONE)
    hardware_background = Column(SQLEnum(BackgroundLevel), nullable=False, default=BackgroundLevel.NONE)
    robotics_experience = Column(SQLEnum(BackgroundLevel), nullable=False, default=BackgroundLevel.NONE)
    personalization_level = Column(SQLEnum(BackgroundLevel), nullable=False, default=BackgroundLevel.BEGINNER)
    preferred_language = Column(String, default="en", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"