from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional
from datetime import timedelta
from ..database import get_db
from ..models.user import User, BackgroundLevel
from ..services.auth_service import AuthService
from ..config import settings
import uuid

router = APIRouter()
security = HTTPBearer()
auth_service = AuthService()


class UserRegistrationRequest:
    def __init__(self, email: str, password: str, first_name: str = None, last_name: str = None,
                 software_background: BackgroundLevel = BackgroundLevel.NONE,
                 hardware_background: BackgroundLevel = BackgroundLevel.NONE,
                 robotics_experience: BackgroundLevel = BackgroundLevel.NONE):
        self.email = email
        self.password = password
        self.first_name = first_name
        self.last_name = last_name
        self.software_background = software_background
        self.hardware_background = hardware_background
        self.robotics_experience = robotics_experience


class UserLoginRequest:
    def __init__(self, email: str, password: str):
        self.email = email
        self.password = password


class UserUpdateRequest:
    def __init__(self, first_name: str = None, last_name: str = None,
                 software_background: BackgroundLevel = None,
                 hardware_background: BackgroundLevel = None,
                 robotics_experience: BackgroundLevel = None,
                 personalization_level: BackgroundLevel = None):
        self.first_name = first_name
        self.last_name = last_name
        self.software_background = software_background
        self.hardware_background = hardware_background
        self.robotics_experience = robotics_experience
        self.personalization_level = personalization_level


@router.post("/register", response_model=dict)
async def register_user(
    email: str,
    password: str,
    first_name: str = None,
    last_name: str = None,
    software_background: BackgroundLevel = BackgroundLevel.NONE,
    hardware_background: BackgroundLevel = BackgroundLevel.NONE,
    robotics_experience: BackgroundLevel = BackgroundLevel.NONE,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists"
        )

    # Create new user
    user = auth_service.create_user(
        db=db,
        email=email,
        password=password,
        first_name=first_name,
        last_name=last_name,
        software_background=software_background,
        hardware_background=hardware_background,
        robotics_experience=robotics_experience
    )

    return {
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "created_at": user.created_at
    }


@router.post("/login", response_model=dict)
async def login_user(
    email: str,
    password: str,
    db: Session = Depends(get_db)
):
    """Authenticate user and return access token."""
    user = auth_service.authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "software_background": user.software_background,
            "hardware_background": user.hardware_background,
            "robotics_experience": user.robotics_experience,
            "personalization_level": user.personalization_level
        }
    }


@router.get("/profile", response_model=dict)
async def get_profile(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current user's profile."""
    token = credentials.credentials
    user = auth_service.get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "id": user.id,
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "software_background": user.software_background,
        "hardware_background": user.hardware_background,
        "robotics_experience": user.robotics_experience,
        "personalization_level": user.personalization_level,
        "created_at": user.created_at,
        "updated_at": user.updated_at
    }


@router.put("/profile", response_model=dict)
async def update_profile(
    first_name: str = None,
    last_name: str = None,
    software_background: BackgroundLevel = None,
    hardware_background: BackgroundLevel = None,
    robotics_experience: BackgroundLevel = None,
    personalization_level: BackgroundLevel = None,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Update user's profile."""
    token = credentials.credentials
    current_user = auth_service.get_current_user(token, db)
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    updated_user = auth_service.update_user_profile(
        db=db,
        user_id=current_user.id,
        first_name=first_name,
        last_name=last_name,
        software_background=software_background,
        hardware_background=hardware_background,
        robotics_experience=robotics_experience,
        personalization_level=personalization_level
    )

    return {
        "id": updated_user.id,
        "email": updated_user.email,
        "first_name": updated_user.first_name,
        "last_name": updated_user.last_name,
        "software_background": updated_user.software_background,
        "hardware_background": updated_user.hardware_background,
        "robotics_experience": updated_user.robotics_experience,
        "personalization_level": updated_user.personalization_level,
        "updated_at": updated_user.updated_at
    }