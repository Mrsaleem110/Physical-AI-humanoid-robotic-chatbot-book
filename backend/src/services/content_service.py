from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import asc
from ..models.chapter import Chapter, Module
from ..models.user import User, BackgroundLevel
from ..models.interaction import UserInteraction, InteractionType
from fastapi import HTTPException, status
import uuid


class ContentService:
    def __init__(self):
        pass

    def get_modules(self, db: Session) -> List[Module]:
        """Get all published modules."""
        return db.query(Module).filter(Module.is_published == True).order_by(asc(Module.order_number)).all()

    def get_module_by_id(self, db: Session, module_id: str) -> Optional[Module]:
        """Get a specific module by ID."""
        return db.query(Module).filter(Module.id == module_id, Module.is_published == True).first()

    def get_module_with_chapters(self, db: Session, module_id: str) -> Optional[dict]:
        """Get a module with all its published chapters."""
        module = db.query(Module).filter(Module.id == module_id, Module.is_published == True).first()
        if not module:
            return None

        chapters = db.query(Chapter).filter(
            Chapter.module_id == module_id,
            Chapter.is_published == True
        ).order_by(asc(Chapter.order_number)).all()

        return {
            "module": module,
            "chapters": chapters
        }

    def get_chapters(self, db: Session, module_id: Optional[str] = None, skip: int = 0, limit: int = 10) -> List[Chapter]:
        """Get all published chapters, optionally filtered by module."""
        query = db.query(Chapter).filter(Chapter.is_published == True)

        if module_id:
            query = query.filter(Chapter.module_id == module_id)

        return query.order_by(asc(Chapter.order_number)).offset(skip).limit(limit).all()

    def get_chapter_by_id(self, db: Session, chapter_id: str) -> Optional[Chapter]:
        """Get a specific chapter by ID."""
        return db.query(Chapter).filter(Chapter.id == chapter_id, Chapter.is_published == True).first()

    def get_chapter_content(self, db: Session, chapter_id: str, user_id: Optional[str] = None,
                           personalization_level: Optional[BackgroundLevel] = None) -> Optional[dict]:
        """Get chapter content, optionally personalized for a user."""
        chapter = self.get_chapter_by_id(db, chapter_id)
        if not chapter:
            return None

        # Record view interaction if user is provided
        if user_id:
            interaction = UserInteraction(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content_id=chapter_id,
                content_type="chapter",
                interaction_type=InteractionType.VIEW
            )
            db.add(interaction)
            db.commit()

        # Determine content to return based on personalization
        content = chapter.content_en  # Default to English content
        if personalization_level:
            # In a real implementation, this would call a personalization service
            # to adapt the content based on the user's level
            content = self.adapt_content_for_level(chapter.content_en, personalization_level)

        return {
            "id": chapter.id,
            "title": chapter.title,
            "content": content,
            "module_id": chapter.module_id,
            "order_number": chapter.order_number,
            "available_languages": ["en", "ur"] if chapter.content_ur else ["en"],
            "is_personalized": personalization_level is not None,
            "personalization_level": personalization_level.value if personalization_level else None,
            "objectives": chapter.objectives,
            "summary": chapter.summary
        }

    def adapt_content_for_level(self, content: str, level: BackgroundLevel) -> str:
        """Adapt content based on user's experience level."""
        # This is a simplified implementation
        # In a real system, this would use AI to adapt content complexity
        if level == BackgroundLevel.BEGINNER:
            # Add more explanations and examples for beginners
            return f"{content}\n\n*This content has been adapted for beginners. More detailed explanations and examples are provided.*"
        elif level == BackgroundLevel.INTERMEDIATE:
            # Standard content for intermediate users
            return content
        elif level == BackgroundLevel.ADVANCED:
            # More concise content for advanced users
            return f"{content}\n\n*This content has been adapted for advanced users. Technical details are provided concisely.*"
        else:
            return content

    def create_module(self, db: Session, title: str, description: str, order_number: int) -> Module:
        """Create a new module."""
        module_id = str(uuid.uuid4())

        db_module = Module(
            id=module_id,
            title=title,
            slug=title.lower().replace(" ", "-").replace("_", "-"),
            description=description,
            order_number=order_number
        )

        db.add(db_module)
        db.commit()
        db.refresh(db_module)
        return db_module

    def create_chapter(self, db: Session, module_id: str, title: str, content_en: str,
                      order_number: int, objectives: str = None, summary: str = None) -> Chapter:
        """Create a new chapter."""
        chapter_id = str(uuid.uuid4())

        db_chapter = Chapter(
            id=chapter_id,
            module_id=module_id,
            title=title,
            slug=title.lower().replace(" ", "-").replace("_", "-"),
            content_en=content_en,
            order_number=order_number,
            objectives=objectives,
            summary=summary
        )

        db.add(db_chapter)
        db.commit()
        db.refresh(db_chapter)
        return db_chapter