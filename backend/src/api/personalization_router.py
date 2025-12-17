"""
Personalization API Router for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional
import logging
import json

from src.models.user import UserResponse
from src.models.personalization import (
    UserPreferenceCreate,
    UserPreferenceUpdate,
    UserPreferenceResponse,
    ContentPersonalizationCreate,
    ContentPersonalizationUpdate,
    ContentPersonalizationResponse
)
from src.services.database import get_db_session
from src.services.auth_service import get_current_user
from src.services.personalization_service import personalization_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
personalization_router = APIRouter(
    prefix="/personalization",
    tags=["personalization"],
    responses={404: {"description": "Not found"}}
)

@personalization_router.get("/preferences", response_model=UserPreferenceResponse)
async def get_user_preferences(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get user preferences for personalization
    """
    try:
        from sqlalchemy.future import select
        from src.models.personalization import UserPreference

        result = await db_session.execute(
            select(UserPreference)
            .where(UserPreference.user_id == current_user.id)
        )
        user_pref = result.scalars().first()

        if not user_pref:
            # Return default preferences if none exist
            return UserPreferenceResponse(
                id=0,
                user_id=current_user.id,
                difficulty_level="intermediate",
                learning_style="visual",
                content_format="mixed",
                update_frequency="daily",
                notification_preferences={},
                adaptive_preferences={},
                enable_personalization=True,
                enable_adaptive_content=True,
                created_at=datetime.utcnow().isoformat(),
                updated_at=None
            )

        return UserPreferenceResponse.from_orm(user_pref)

    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.post("/preferences", response_model=UserPreferenceResponse)
async def set_user_preferences(
    preferences: UserPreferenceCreate,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Set user preferences for personalization
    """
    try:
        from sqlalchemy.future import select
        from src.models.personalization import UserPreference

        # Check if preferences already exist
        result = await db_session.execute(
            select(UserPreference)
            .where(UserPreference.user_id == current_user.id)
        )
        existing_pref = result.scalars().first()

        if existing_pref:
            # Update existing preferences
            for field, value in preferences.dict().items():
                if hasattr(existing_pref, field):
                    setattr(existing_pref, field, value)
            updated_pref = existing_pref
        else:
            # Create new preferences
            new_pref = UserPreference(
                user_id=current_user.id,
                difficulty_level=preferences.difficulty_level,
                learning_style=preferences.learning_style,
                content_format=preferences.content_format,
                update_frequency=preferences.update_frequency,
                notification_preferences=json.dumps(preferences.notification_preferences),
                adaptive_preferences=json.dumps(preferences.adaptive_preferences),
                enable_personalization=preferences.enable_personalization,
                enable_adaptive_content=preferences.enable_adaptive_content
            )
            db_session.add(new_pref)
            updated_pref = new_pref

        await db_session.commit()
        await db_session.refresh(updated_pref)

        # Also update in personalization service
        success = await personalization_service.update_user_preferences(
            db_session, current_user.id, preferences.dict()
        )

        return UserPreferenceResponse.from_orm(updated_pref)

    except Exception as e:
        logger.error(f"Error setting user preferences: {e}")
        await db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.put("/preferences", response_model=UserPreferenceResponse)
async def update_user_preferences(
    preferences: UserPreferenceUpdate,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Update user preferences for personalization
    """
    try:
        from sqlalchemy.future import select
        from src.models.personalization import UserPreference

        result = await db_session.execute(
            select(UserPreference)
            .where(UserPreference.user_id == current_user.id)
        )
        user_pref = result.scalars().first()

        if not user_pref:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User preferences not found"
            )

        # Update fields that are provided
        for field, value in preferences.dict(exclude_unset=True).items():
            if hasattr(user_pref, field) and value is not None:
                setattr(user_pref, field, value)

        await db_session.commit()
        await db_session.refresh(user_pref)

        return UserPreferenceResponse.from_orm(user_pref)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        await db_session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/personalized-content")
async def get_personalized_content(
    content_type: str = "chapter",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get personalized content based on user preferences
    """
    try:
        personalized_content = await personalization_service.get_personalized_content(
            db_session, current_user.id, content_type
        )

        return {
            "user_id": current_user.id,
            "content_type": content_type,
            "personalized_content": personalized_content,
            "count": len(personalized_content),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting personalized content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/learning-path")
async def get_learning_path(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get personalized learning path for the user
    """
    try:
        learning_path = await personalization_service.get_learning_path(
            db_session, current_user.id
        )

        return {
            "user_id": current_user.id,
            "learning_path": learning_path,
            "path_length": len(learning_path),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting learning path: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/adaptive-content/{content_id}")
async def get_adaptive_content(
    content_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get adaptive content based on user interaction patterns
    """
    try:
        adaptive_content = await personalization_service.get_adaptive_content(
            db_session, current_user.id, content_id
        )

        return {
            "user_id": current_user.id,
            "content_id": content_id,
            "adaptive_content": adaptive_content,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting adaptive content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/recommendations")
async def get_recommendations(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get personalized content recommendations
    """
    try:
        # This would analyze user behavior and preferences to generate recommendations
        # For now, we'll use the learning path as recommendations
        learning_path = await personalization_service.get_learning_path(
            db_session, current_user.id
        )

        # Get top 5 recommendations
        recommendations = learning_path[:5]

        return {
            "user_id": current_user.id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/analytics")
async def get_personalization_analytics(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get analytics about personalization effectiveness
    """
    try:
        # This would return analytics about how well personalization is working
        # For now, return placeholder data
        from datetime import timedelta

        start_date = datetime.utcnow() - timedelta(days=30)

        analytics = {
            "user_id": current_user.id,
            "period": "last_30_days",
            "engagement_metrics": {
                "pages_read": 25,
                "time_spent": "3h 45m",
                "completion_rate": 0.78,
                "content_interactions": 120
            },
            "personalization_metrics": {
                "content_relevance_score": 0.85,
                "preference_accuracy": 0.92,
                "adaptive_learning_improvement": 0.15
            },
            "recommendation_metrics": {
                "click_through_rate": 0.45,
                "conversion_rate": 0.32,
                "satisfaction_score": 4.2
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        return analytics

    except Exception as e:
        logger.error(f"Error getting personalization analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/difficulty-assessment")
async def get_difficulty_assessment(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get difficulty assessment based on user performance
    """
    try:
        # This would analyze user's performance to assess appropriate difficulty
        # For now, return placeholder data based on user preferences
        user_prefs = await personalization_service.get_user_preferences(db_session, current_user.id)

        assessment = {
            "user_id": current_user.id,
            "current_level": user_prefs.get("difficulty_level", "intermediate"),
            "recommended_level": user_prefs.get("difficulty_level", "intermediate"),
            "performance_indicators": {
                "average_completion_time": "25 min",
                "success_rate": 0.85,
                "engagement_level": "high"
            },
            "suggested_adjustments": [],
            "timestamp": datetime.utcnow().isoformat()
        }

        return assessment

    except Exception as e:
        logger.error(f"Error getting difficulty assessment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@personalization_router.get("/health")
async def personalization_health():
    """
    Personalization service health check
    """
    return {
        "status": "healthy",
        "service": "personalization",
        "timestamp": datetime.utcnow().isoformat()
    }


__all__ = ["personalization_router"]