from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, List
from ..database import get_db
from ..models.user import User, BackgroundLevel
from ..models.chapter import Chapter, Module
from ..services.content_service import ContentService
from ..services.auth_service import AuthService

router = APIRouter()
security = HTTPBearer()
content_service = ContentService()
auth_service = AuthService()


@router.get("/modules", response_model=List[dict])
async def get_all_modules(
    db: Session = Depends(get_db)
):
    """Get all published modules."""
    modules = content_service.get_modules(db)
    return [
        {
            "id": module.id,
            "title": module.title,
            "slug": module.slug,
            "description": module.description,
            "order_number": module.order_number,
            "created_at": module.created_at
        }
        for module in modules
    ]


@router.get("/modules/{module_id}", response_model=dict)
async def get_module(
    module_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific module with its chapters."""
    result = content_service.get_module_with_chapters(db, module_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Module not found"
        )

    module = result["module"]
    chapters = result["chapters"]

    return {
        "id": module.id,
        "title": module.title,
        "slug": module.slug,
        "description": module.description,
        "order_number": module.order_number,
        "created_at": module.created_at,
        "updated_at": module.updated_at,
        "chapters": [
            {
                "id": chapter.id,
                "title": chapter.title,
                "slug": chapter.slug,
                "order_number": chapter.order_number,
                "module_id": chapter.module_id
            }
            for chapter in chapters
        ]
    }


@router.get("/chapters", response_model=List[dict])
async def get_all_chapters(
    module_id: Optional[str] = Query(None, description="Filter by module ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get all published chapters with optional module filter."""
    chapters = content_service.get_chapters(db, module_id, skip, limit)
    return [
        {
            "id": chapter.id,
            "title": chapter.title,
            "slug": chapter.slug,
            "module_id": chapter.module_id,
            "order_number": chapter.order_number,
            "objectives": chapter.objectives,
            "summary": chapter.summary
        }
        for chapter in chapters
    ]


@router.get("/chapters/{chapter_id}", response_model=dict)
async def get_chapter(
    chapter_id: str,
    personalization_level: Optional[BackgroundLevel] = Query(None, description="Personalization level for content adaptation"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """Get a specific chapter with content."""
    user_id = None
    if credentials:
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if user:
            user_id = user.id
            # If no personalization level specified, use user's profile level
            if personalization_level is None:
                personalization_level = user.personalization_level

    chapter_content = content_service.get_chapter_content(
        db, chapter_id, user_id, personalization_level
    )
    if not chapter_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    return chapter_content


@router.post("/chapters/{chapter_id}/personalize", response_model=dict)
async def personalize_chapter(
    chapter_id: str,
    level: BackgroundLevel,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get personalized version of chapter content based on user profile."""
    token = credentials.credentials
    user = auth_service.get_current_user(token, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    chapter_content = content_service.get_chapter_content(
        db, chapter_id, user.id, level
    )
    if not chapter_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    return chapter_content