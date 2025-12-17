"""
Robotics Models for Physical AI & Humanoid Robotics Platform
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, relationship
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

Base = declarative_base()

class RoboticsTask(Base):
    """
    Model for robotics tasks and operations
    """
    __tablename__ = "robotics_tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, nullable=False)  # e.g., "navigation", "manipulation", "perception"
    task_description = Column(Text)  # Detailed description of the task
    task_type = Column(String, default="general")  # navigation, manipulation, perception, etc.

    # Task execution details
    robot_id = Column(String, nullable=False)  # ID of the robot executing the task
    task_status = Column(String, default="pending")  # pending, executing, completed, failed, cancelled
    priority = Column(Integer, default=1)  # Priority level (1-10)

    # Task parameters and results
    parameters = Column(JSON)  # Task-specific parameters
    results = Column(JSON)  # Results of task execution
    error_message = Column(Text)  # Error message if task failed

    # Execution details
    estimated_duration = Column(Float)  # Estimated time in seconds
    actual_duration = Column(Float)  # Actual time taken in seconds
    success_rate = Column(Float)  # Success rate for repeated tasks

    # Safety and constraints
    safety_critical = Column(Boolean, default=False)  # Whether task is safety-critical
    requires_human_supervision = Column(Boolean, default=False)  # Whether human supervision required

    # Associated user (if task was initiated by a user)
    user_id = Column(Integer, ForeignKey("users.id"))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True))  # When task execution started
    completed_at = Column(DateTime(timezone=True))  # When task execution completed

    # Relationships
    user = relationship("User", back_populates="robotics_tasks")


class RobotState(Base):
    """
    Model for storing robot state information
    """
    __tablename__ = "robot_states"

    id = Column(Integer, primary_key=True, index=True)
    robot_id = Column(String, nullable=False, index=True)  # Unique identifier for the robot

    # Position and orientation (pose)
    position = Column(JSON)  # {"x": float, "y": float, "z": float}
    orientation = Column(JSON)  # {"x": float, "y": float, "z": float, "w": float} (quaternion)

    # Joint states
    joints = Column(JSON)  # {"joint_name": position, ...}

    # Sensors and status
    battery_level = Column(Float)  # Battery level percentage (0.0 - 100.0)
    status = Column(String, default="idle")  # idle, moving, manipulating, charging, error
    operational = Column(Boolean, default=True)  # Whether robot is operational

    # Environmental data
    environment_map = Column(Text)  # Serialized environment map
    detected_objects = Column(JSON)  # List of detected objects

    # Safety information
    safety_status = Column(String, default="normal")  # normal, warning, danger, emergency
    last_safety_check = Column(DateTime(timezone=True))  # Timestamp of last safety check

    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    robot_tasks = relationship("RoboticsTask", back_populates="robot_state")


# Add relationship to User model
from src.models.user import User
User.robotics_tasks = relationship("RoboticsTask", back_populates="user")


class RoboticsTaskCreate(BaseModel):
    """Schema for creating a robotics task"""
    task_name: str
    task_description: Optional[str] = None
    task_type: Optional[str] = "general"
    robot_id: str
    parameters: Optional[Dict[str, Any]] = {}
    priority: Optional[int] = 1
    estimated_duration: Optional[float] = None
    safety_critical: Optional[bool] = False
    requires_human_supervision: Optional[bool] = False
    user_id: Optional[int] = None


class RoboticsTaskUpdate(BaseModel):
    """Schema for updating a robotics task"""
    task_status: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class RoboticsTaskResponse(BaseModel):
    """Schema for robotics task response"""
    id: int
    task_name: str
    task_description: Optional[str]
    task_type: str
    robot_id: str
    task_status: str
    priority: int
    parameters: Optional[Dict[str, Any]]
    results: Optional[Dict[str, Any]]
    error_message: Optional[str]
    estimated_duration: Optional[float]
    actual_duration: Optional[float]
    success_rate: Optional[float]
    safety_critical: bool
    requires_human_supervision: bool
    user_id: Optional[int]
    created_at: str
    updated_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]

    class Config:
        from_attributes = True


class RobotStateCreate(BaseModel):
    """Schema for creating robot state"""
    robot_id: str
    position: Optional[Dict[str, float]] = {}
    orientation: Optional[Dict[str, float]] = {}
    joints: Optional[Dict[str, float]] = {}
    battery_level: Optional[float] = None
    status: Optional[str] = "idle"
    operational: Optional[bool] = True
    environment_map: Optional[str] = None
    detected_objects: Optional[Dict[str, Any]] = {}
    safety_status: Optional[str] = "normal"


class RobotStateResponse(BaseModel):
    """Schema for robot state response"""
    id: int
    robot_id: str
    position: Optional[Dict[str, Any]]
    orientation: Optional[Dict[str, Any]]
    joints: Optional[Dict[str, Any]]
    battery_level: Optional[float]
    status: str
    operational: bool
    environment_map: Optional[str]
    detected_objects: Optional[Dict[str, Any]]
    safety_status: str
    timestamp: str

    class Config:
        from_attributes = True


class RoboticsCommand(BaseModel):
    """Schema for robotics command request"""
    robot_id: str
    command: str  # The command to execute
    parameters: Dict[str, Any]  # Command parameters
    priority: int = 1  # Command priority
    timeout: float = 30.0  # Timeout in seconds
    requires_confirmation: bool = False  # Whether command requires user confirmation


class RoboticsResponse(BaseModel):
    """Schema for robotics command response"""
    robot_id: str
    command: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str


class SubagentDefinition(BaseModel):
    """Schema for defining Claude Code Subagents"""
    name: str
    description: str
    capabilities: List[str]  # List of capabilities this subagent has
    skills: List[str]  # List of skills this subagent can perform
    parameters: Dict[str, Any]  # Default parameters for the subagent
    enabled: bool = True


class SubagentExecutionRequest(BaseModel):
    """Schema for executing a subagent task"""
    subagent_name: str
    task: str
    parameters: Dict[str, Any]
    timeout: float = 60.0  # Timeout in seconds


class SubagentExecutionResponse(BaseModel):
    """Schema for subagent execution response"""
    subagent_name: str
    task: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    success: bool
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str