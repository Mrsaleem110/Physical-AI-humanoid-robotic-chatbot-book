"""
Robotics API Router for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime

from src.models.user import UserResponse
from src.models.robotics import (
    RoboticsCommand,
    RoboticsResponse,
    SubagentDefinition,
    SubagentExecutionRequest,
    SubagentExecutionResponse
)
from src.services.database import get_db_session
from src.services.auth_service import get_current_user
from src.services.robotics_service import robotics_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
robotics_router = APIRouter(
    prefix="/robotics",
    tags=["robotics"],
    responses={404: {"description": "Not found"}}
)

@robotics_router.post("/command", response_model=RoboticsResponse)
async def execute_robotics_command(
    command: RoboticsCommand,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Execute a robotics command using Claude Code Subagents
    """
    try:
        # Execute the command using robotics service
        result = await robotics_service.execute_robotics_command(
            command=command.command,
            parameters=command.parameters
        )

        return RoboticsResponse(
            robot_id=command.robot_id,
            command=command.command,
            success=result["success"],
            result=result.get("result"),
            error=result.get("error"),
            execution_time=result.get("execution_time"),
            timestamp=result["timestamp"]
        )

    except Exception as e:
        logger.error(f"Error executing robotics command: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Robotics command execution error"
        )


@robotics_router.post("/subagent/execute", response_model=SubagentExecutionResponse)
async def execute_subagent_task(
    request: SubagentExecutionRequest,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Execute a task using Claude Code Subagent
    """
    try:
        # Execute the subagent task
        result = await robotics_service.execute_subagent_task(
            subagent_name=request.subagent_name,
            task=request.task,
            parameters=request.parameters
        )

        return SubagentExecutionResponse(
            subagent_name=request.subagent_name,
            task=request.task,
            parameters=request.parameters,
            result=result.get("result"),
            success=result.get("success", False),
            error=result.get("error"),
            execution_time=result.get("execution_time"),
            timestamp=result["timestamp"]
        )

    except Exception as e:
        logger.error(f"Error executing subagent task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Subagent execution error"
        )


@robotics_router.get("/subagents")
async def get_available_subagents():
    """
    Get list of available Claude Code Subagents
    """
    try:
        subagents = await robotics_service.get_available_subagents()
        return {
            "subagents": subagents,
            "count": len(subagents),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting subagents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving subagents"
        )


@robotics_router.get("/subagents/{subagent_name}")
async def get_subagent_capabilities(subagent_name: str):
    """
    Get capabilities of a specific subagent
    """
    try:
        capabilities = await robotics_service.get_subagent_capabilities(subagent_name)
        return capabilities

    except Exception as e:
        logger.error(f"Error getting subagent capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving subagent capabilities"
        )


@robotics_router.post("/subagent/define")
async def define_subagent(
    definition: SubagentDefinition,
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Define a new Claude Code Subagent
    """
    try:
        # Check if user is admin
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required to define subagents"
            )

        # Initialize the subagent
        await robotics_service.initialize_subagent(definition.name)

        return {
            "name": definition.name,
            "description": definition.description,
            "capabilities": definition.capabilities,
            "skills": definition.skills,
            "enabled": definition.enabled,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error defining subagent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error defining subagent"
        )


@robotics_router.post("/complex-task")
async def execute_complex_task(
    task_description: str,
    parameters: Dict = {},
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Execute a complex task that may require multiple subagents
    """
    try:
        result = await robotics_service.execute_complex_task(
            task_description=task_description,
            parameters=parameters
        )

        return result

    except Exception as e:
        logger.error(f"Error executing complex task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing complex task"
        )


@robotics_router.get("/robot-state/{robot_id}")
async def get_robot_state(
    robot_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get current state of a robot
    """
    try:
        state = await robotics_service.get_robot_state(db_session, robot_id)
        return state

    except Exception as e:
        logger.error(f"Error getting robot state: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving robot state"
        )


@robotics_router.get("/robot-states")
async def get_all_robot_states(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get states of all robots
    """
    try:
        # This would return states of all robots
        # For now, return a placeholder
        return {
            "robot_states": [],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting robot states: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving robot states"
        )


@robotics_router.post("/navigation/move-to")
async def move_to_location(
    robot_id: str,
    x: float,
    y: float,
    theta: float = 0.0,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Move robot to a specific location
    """
    try:
        parameters = {"x": x, "y": y, "theta": theta}

        result = await robotics_service.execute_subagent_task(
            subagent_name="navigation_subagent",
            task="move_to",
            parameters=parameters
        )

        return result

    except Exception as e:
        logger.error(f"Error in move_to command: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing move_to command"
        )


@robotics_router.post("/manipulation/grasp-object")
async def grasp_object(
    robot_id: str,
    object_name: str,
    grasp_type: str = "pinch",
    force: float = 10.0,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Grasp an object with the robot
    """
    try:
        parameters = {
            "object": object_name,
            "grasp_type": grasp_type,
            "force": force
        }

        result = await robotics_service.execute_subagent_task(
            subagent_name="manipulation_subagent",
            task="grasp_object",
            parameters=parameters
        )

        return result

    except Exception as e:
        logger.error(f"Error in grasp_object command: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing grasp_object command"
        )


@robotics_router.post("/perception/detect-objects")
async def detect_objects(
    robot_id: str,
    object_type: str = "any",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Detect objects in the robot's environment
    """
    try:
        parameters = {"type": object_type}

        result = await robotics_service.execute_subagent_task(
            subagent_name="perception_subagent",
            task="detect_objects",
            parameters=parameters
        )

        return result

    except Exception as e:
        logger.error(f"Error in detect_objects command: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing detect_objects command"
        )


@robotics_router.post("/interaction/speak")
async def speak_text(
    robot_id: str,
    text: str,
    language: str = "en",
    volume: float = 0.8,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Make the robot speak text
    """
    try:
        parameters = {
            "text": text,
            "language": language,
            "volume": volume
        }

        result = await robotics_service.execute_subagent_task(
            subagent_name="interaction_subagent",
            task="speak",
            parameters=parameters
        )

        return result

    except Exception as e:
        logger.error(f"Error in speak command: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing speak command"
        )


@robotics_router.get("/skills")
async def get_available_skills():
    """
    Get list of all available skills across subagents
    """
    try:
        # Return the skill registry
        skills = {}
        for category, skill_dict in robotics_service.skill_registry.items():
            skills[category] = list(skill_dict.keys())

        return {
            "skills_by_category": skills,
            "total_categories": len(skills),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting available skills: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving available skills"
        )


@robotics_router.get("/health")
async def robotics_health():
    """
    Robotics service health check
    """
    return {
        "status": "healthy",
        "service": "robotics",
        "subagents_initialized": len(await robotics_service.get_available_subagents()),
        "timestamp": datetime.utcnow().isoformat()
    }


__all__ = ["robotics_router"]