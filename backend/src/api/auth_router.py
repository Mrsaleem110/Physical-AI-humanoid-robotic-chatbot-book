"""
Authentication API Router for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from src.models.user import UserCreate, UserUpdate, UserResponse
from src.services.auth_service import (
    auth_service,
    get_user_by_email,
    create_user,
    update_user,
    authenticate_user,
    get_user_by_id
)
from src.services.database import get_db_session
from src.utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Create router
auth_router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={404: {"description": "Not found"}}
)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db_session: AsyncSession = Depends(get_db_session)
) -> UserResponse:
    """
    Get current authenticated user
    """
    user = await auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserResponse.from_orm(user)

@auth_router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Register a new user
    """
    try:
        # Check if user already exists
        existing_user = await get_user_by_email(db_session, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )

        # Create new user
        user = await create_user(
            db_session,
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            software_background=user_data.software_background,
            hardware_background=user_data.hardware_background,
            robotics_experience=user_data.robotics_experience
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user"
            )

        return UserResponse.from_orm(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@auth_router.post("/login")
async def login_user(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Login user and return access token
    """
    try:
        # Authenticate user
        token, user = await auth_service.login(form_data.username, form_data.password)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": UserResponse.from_orm(user)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@auth_router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    """
    Get current user's information
    """
    return current_user


@auth_router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    current_user: UserResponse = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Update current user's information
    """
    try:
        # Update user
        updated_user = await update_user(
            db_session,
            current_user.id,
            first_name=user_update.first_name,
            last_name=user_update.last_name,
            bio=user_update.bio,
            software_background=user_update.software_background,
            hardware_background=user_update.hardware_background,
            robotics_experience=user_update.robotics_experience
        )

        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        return UserResponse.from_orm(updated_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@auth_router.post("/change-password")
async def change_password(
    request: Request,
    current_user: UserResponse = Depends(get_current_user),
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Change user password
    """
    try:
        # Get password data from request
        body = await request.json()
        old_password = body.get("old_password")
        new_password = body.get("new_password")

        if not old_password or not new_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both old and new passwords are required"
            )

        success = await auth_service.change_password(current_user.id, old_password, new_password)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Old password is incorrect"
            )

        return {"detail": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@auth_router.get("/health")
async def auth_health():
    """
    Authentication service health check
    """
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


# Additional utility functions
async def require_admin_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Dependency to require admin user
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


@auth_router.get("/admin/users", dependencies=[Depends(require_admin_user)])
async def get_all_users(
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Get all users (admin only)
    """
    # This would implement getting all users for admin panel
    # Implementation would go here
    pass


__all__ = ["auth_router"]