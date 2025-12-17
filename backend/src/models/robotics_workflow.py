from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from .user import Base

Base = declarative_base()


class RoboticsWorkflow(Base):
    __tablename__ = "robotics_workflows"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    workflow_definition = Column(JSON, nullable=False)  # Full workflow definition
    subagent_config = Column(JSON, nullable=False)  # Subagent configuration
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<RoboticsWorkflow(id={self.id}, name={self.name})>"