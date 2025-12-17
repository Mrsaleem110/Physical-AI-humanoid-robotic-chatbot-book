from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Dict
from ..database import get_db
from ..models.user import User, BackgroundLevel
from ..services.personalization_service import PersonalizationService
from ..services.auth_service import AuthService

router = APIRouter()
security = HTTPBearer()
personalization_service = PersonalizationService()
auth_service = AuthService()


@router.get("/profile", response_model=Dict[str, str])
async def get_personalization_profile(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get user's personalization profile."""
    token = credentials.credentials
    user = auth_service.get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    profile = personalization_service.get_user_profile(db, user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found"
        )

    return profile


@router.post("/level", response_model=Dict[str, str])
async def set_personalization_level(
    level: BackgroundLevel,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Update user's personalization level."""
    token = credentials.credentials
    user = auth_service.get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    updated_user = personalization_service.update_user_personalization_level(
        db, user.id, level
    )

    return {
        "id": updated_user.id,
        "email": updated_user.email,
        "personalization_level": updated_user.personalization_level.value,
        "updated_at": updated_user.updated_at
    }


@router.get("/recommended", response_model=List[Dict[str, str]])
async def get_recommended_content(
    current_chapter_id: str = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get recommended content based on user's profile and progress."""
    token = credentials.credentials
    user = auth_service.get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    recommendations = personalization_service.get_recommended_content(
        db, user.id, current_chapter_id
    )

    return recommendations


@router.get("/path", response_model=List[Dict[str, str]])
async def get_learning_path(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get a personalized learning path for the user."""
    token = credentials.credentials
    user = auth_service.get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    path = personalization_service.get_learning_path(db, user.id)

    return path