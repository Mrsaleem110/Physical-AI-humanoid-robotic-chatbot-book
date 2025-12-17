from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from .user import Base

Base = declarative_base()


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(String, primary_key=True, index=True)
    module_id = Column(String, ForeignKey("modules.id"), nullable=False)
    title = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)
    content_en = Column(Text, nullable=False)
    content_ur = Column(Text, nullable=True)  # Urdu translation
    order_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    is_published = Column(Boolean, default=True, nullable=False)
    objectives = Column(Text, nullable=True)  # Learning objectives
    summary = Column(Text, nullable=True)     # Chapter summary

    def __repr__(self):
        return f"<Chapter(id={self.id}, title={self.title})>"


class Module(Base):
    __tablename__ = "modules"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    slug = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    order_number = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    is_published = Column(Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<Module(id={self.id}, title={self.title})>"