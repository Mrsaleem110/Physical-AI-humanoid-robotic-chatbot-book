"""
Content API Router for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from src.models.chapter import ModuleCreate, ChapterCreate, ModuleResponse, ChapterResponse
from src.models.user import UserResponse
from src.services.database import get_db_session
from src.services.auth_service import get_current_user
from src.services.personalization_service import personalization_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
content_router = APIRouter(
    prefix="/content",
    tags=["content"],
    responses={404: {"description": "Not found"}}
)

@content_router.get("/modules", response_model=List[ModuleResponse])
async def get_modules(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get all available modules
    """
    try:
        # This would fetch modules from the database
        # For now, we'll return an empty list
        # In a real implementation, this would query the database
        from sqlalchemy.future import select
        from src.models.chapter import Module

        result = await db_session.execute(
            select(Module)
            .where(Module.is_active == True)
            .order_by(Module.position)
        )
        modules = result.scalars().all()

        return [ModuleResponse.from_orm(m) for m in modules]

    except Exception as e:
        logger.error(f"Error getting modules: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/modules/{module_id}", response_model=ModuleResponse)
async def get_module(
    module_id: int,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get a specific module by ID
    """
    try:
        from sqlalchemy.future import select
        from src.models.chapter import Module

        result = await db_session.execute(
            select(Module)
            .where(Module.id == module_id, Module.is_active == True)
        )
        module = result.scalar_one_or_none()

        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module not found"
            )

        return ModuleResponse.from_orm(module)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting module {module_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/chapters", response_model=List[ChapterResponse])
async def get_chapters(
    module_id: Optional[int] = None,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get all chapters, optionally filtered by module
    """
    try:
        from sqlalchemy.future import select
        from src.models.chapter import Chapter

        query = select(Chapter).where(Chapter.is_active == True)
        if module_id:
            query = query.where(Chapter.module_id == module_id)

        query = query.order_by(Chapter.position)

        result = await db_session.execute(query)
        chapters = result.scalars().all()

        return [ChapterResponse.from_orm(c) for c in chapters]

    except Exception as e:
        logger.error(f"Error getting chapters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/chapters/{chapter_id}", response_model=ChapterResponse)
async def get_chapter(
    chapter_id: int,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get a specific chapter by ID
    """
    try:
        from sqlalchemy.future import select
        from src.models.chapter import Chapter

        result = await db_session.execute(
            select(Chapter)
            .where(Chapter.id == chapter_id, Chapter.is_active == True)
        )
        chapter = result.scalar_one_or_none()

        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chapter not found"
            )

        return ChapterResponse.from_orm(chapter)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chapter {chapter_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/personalized-content")
async def get_personalized_content(
    content_type: str = "chapter",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get personalized content based on user preferences
    """
    try:
        # Get personalized content using personalization service
        personalized_content = await personalization_service.get_personalized_content(
            db_session, current_user.id, content_type
        )

        return {
            "content_type": content_type,
            "personalized_content": personalized_content,
            "user_id": current_user.id,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting personalized content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/learning-path")
async def get_learning_path(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get personalized learning path for the user
    """
    try:
        # Get personalized learning path using personalization service
        learning_path = await personalization_service.get_learning_path(
            db_session, current_user.id
        )

        return {
            "learning_path": learning_path,
            "user_id": current_user.id,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting learning path: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/search")
async def search_content(
    query: str,
    content_type: Optional[str] = None,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Search for content based on query
    """
    try:
        # This would implement search functionality
        # For now, we'll return an empty result
        # In a real implementation, this would search the database or vector store
        from sqlalchemy.future import select
        from src.models.chapter import Chapter, Module

        # Build search query based on content type
        if content_type == "chapter":
            search_query = select(Chapter).where(
                Chapter.is_active == True,
                (Chapter.title.contains(query)) | (Chapter.content.contains(query))
            )
        elif content_type == "module":
            search_query = select(Module).where(
                Module.is_active == True,
                (Module.title.contains(query)) | (Module.description.contains(query))
            )
        else:
            # Search both chapters and modules
            chapters_query = select(Chapter).where(
                Chapter.is_active == True,
                (Chapter.title.contains(query)) | (Chapter.content.contains(query))
            )

            modules_query = select(Module).where(
                Module.is_active == True,
                (Module.title.contains(query)) | (Module.description.contains(query))
            )

            # Execute both queries
            chapters_result = await db_session.execute(chapters_query)
            modules_result = await db_session.execute(modules_query)

            chapters = [ChapterResponse.from_orm(c) for c in chapters_result.scalars().all()]
            modules = [ModuleResponse.from_orm(m) for m in modules_result.scalars().all()]

            return {
                "query": query,
                "results": {
                    "chapters": chapters,
                    "modules": modules
                },
                "total_results": len(chapters) + len(modules),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

        # Execute query
        result = await db_session.execute(search_query)
        results = result.scalars().all()

        if content_type == "chapter":
            return {
                "query": query,
                "results": [ChapterResponse.from_orm(c) for c in results],
                "total_results": len(results),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }
        else:
            return {
                "query": query,
                "results": [ModuleResponse.from_orm(m) for m in results],
                "total_results": len(results),
                "timestamp": __import__('datetime').datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Error searching content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/progress/{user_id}")
async def get_user_progress(
    user_id: int,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get user's progress through content
    """
    try:
        # Check if user is requesting their own progress or is admin
        if current_user.id != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this user's progress"
            )

        # This would implement progress tracking
        # For now, we'll return a placeholder response
        return {
            "user_id": user_id,
            "progress": {
                "total_modules": 4,
                "completed_modules": 1,
                "total_chapters": 16,
                "completed_chapters": 3,
                "completion_percentage": 18.75,
                "time_spent": "2h 30m",
                "last_accessed": __import__('datetime').datetime.utcnow().isoformat()
            },
            "timestamp": __import__('datetime').datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@content_router.get("/health")
async def content_health():
    """
    Content service health check
    """
    return {
        "status": "healthy",
        "service": "content",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }


__all__ = ["content_router"]