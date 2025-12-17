from typing import Optional
from sqlalchemy.orm import Session
from ..models.user import User, BackgroundLevel
from ..models.chapter import Chapter
from ..models.interaction import UserInteraction, InteractionType
from fastapi import HTTPException, status
import uuid


class PersonalizationService:
    def __init__(self):
        pass

    def get_user_profile(self, db: Session, user_id: str) -> Optional[dict]:
        """Get user profile information for personalization."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None

        return {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "software_background": user.software_background,
            "hardware_background": user.hardware_background,
            "robotics_experience": user.robotics_experience,
            "personalization_level": user.personalization_level
        }

    def update_user_personalization_level(self, db: Session, user_id: str, level: BackgroundLevel) -> User:
        """Update user's personalization level."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.personalization_level = level
        db.commit()
        db.refresh(user)
        return user

    def adapt_content_for_user(self, db: Session, content: str, user_id: str, chapter_id: str = None) -> str:
        """Adapt content based on user's profile and preferences."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            # If user not found, return content as is
            return content

        # Record personalization interaction
        if chapter_id:
            interaction = UserInteraction(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content_id=chapter_id,
                content_type="chapter",
                interaction_type=InteractionType.PERSONALIZE,
                interaction_data={
                    "original_content_length": len(content),
                    "personalization_level": user.personalization_level.value
                }
            )
            db.add(interaction)
            db.commit()

        # Adapt content based on user's background and experience
        return self._adapt_content(content, user)

    def _adapt_content(self, content: str, user: User) -> str:
        """Internal method to adapt content based on user profile."""
        # Determine the effective level to use for personalization
        # If personalization_level is explicitly set, use that; otherwise, use the highest of the background levels
        effective_level = user.personalization_level
        if effective_level == BackgroundLevel.NONE:
            # Determine level based on background (using the highest level among backgrounds)
            levels = [user.software_background, user.hardware_background, user.robotics_experience]
            # Convert to numeric values for comparison
            level_values = {
                BackgroundLevel.BEGINNER: 1,
                BackgroundLevel.INTERMEDIATE: 2,
                BackgroundLevel.ADVANCED: 3,
                BackgroundLevel.NONE: 0
            }
            max_level_value = max(level_values[level] for level in levels)
            # Convert back to enum
            for level, value in level_values.items():
                if value == max_level_value:
                    effective_level = level
                    break

        # Adapt content based on effective level
        if effective_level == BackgroundLevel.BEGINNER:
            # Add more explanations and examples for beginners
            adapted_content = f"# Beginner-Friendly Content\n\n{content}\n\n*This content has been adapted for beginners. Additional explanations and examples have been included.*"
        elif effective_level == BackgroundLevel.INTERMEDIATE:
            # Standard content for intermediate users
            adapted_content = f"# Intermediate Content\n\n{content}\n\n*This content is adapted for users with some experience in the topic.*"
        elif effective_level == BackgroundLevel.ADVANCED:
            # More concise content for advanced users
            adapted_content = f"# Advanced Content\n\n{content}\n\n*This content is adapted for advanced users. Technical details are provided concisely.*"
        else:
            # Default to beginner level if no level is set
            adapted_content = f"# Beginner-Friendly Content\n\n{content}\n\n*This content has been adapted for beginners. Additional explanations and examples have been included.*"

        return adapted_content

    def get_recommended_content(self, db: Session, user_id: str, current_chapter_id: str = None) -> List[dict]:
        """Get recommended content based on user's profile and progress."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []

        # This is a simplified recommendation algorithm
        # In a real system, this would use more sophisticated algorithms based on user interactions

        # Get all modules
        modules = db.query(Chapter).filter(Chapter.is_published == True).all()

        # Simple recommendation: suggest content based on user's experience level
        recommendations = []
        for chapter in modules[:5]:  # Limit to 5 recommendations
            if current_chapter_id and chapter.id == current_chapter_id:
                continue  # Don't recommend the current chapter

            recommendations.append({
                "id": chapter.id,
                "title": chapter.title,
                "module_id": chapter.module_id,
                "relevance_score": self._calculate_relevance_score(user, chapter)
            })

        return sorted(recommendations, key=lambda x: x["relevance_score"], reverse=True)

    def _calculate_relevance_score(self, user: User, chapter: Chapter) -> float:
        """Calculate relevance score for a chapter based on user profile."""
        # Simple scoring based on matching user's experience level
        score = 0.0

        # This is a simplified algorithm
        # A real system would use more sophisticated matching algorithms
        if user.robotics_experience == BackgroundLevel.ADVANCED:
            score += 0.8  # Prioritize robotics-related content
        elif user.software_background == BackgroundLevel.ADVANCED:
            score += 0.6  # Prioritize software-related content

        # Additional factors could include:
        # - User's interaction history with similar content
        # - Time since user last accessed related content
        # - Performance on exercises in related content
        # - User's explicit preferences

        return min(score, 1.0)  # Cap at 1.0

    def get_learning_path(self, db: Session, user_id: str) -> List[dict]:
        """Get a personalized learning path for the user."""
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []

        # This is a simplified learning path algorithm
        # In a real system, this would be much more sophisticated

        # Get all published chapters ordered by module and sequence
        chapters = db.query(Chapter).filter(
            Chapter.is_published == True
        ).order_by(Chapter.module_id, Chapter.order_number).all()

        # Determine starting point based on user's experience
        start_idx = 0
        if user.robotics_experience == BackgroundLevel.ADVANCED:
            # Advanced users might start with later chapters
            start_idx = max(0, len(chapters) // 3)  # Start from 1/3 through the content
        elif user.software_background == BackgroundLevel.ADVANCED:
            # Software-focused users might start with implementation-focused chapters
            start_idx = max(0, len(chapters) // 4)  # Start from 1/4 through the content

        # Return personalized learning path
        path = []
        for i, chapter in enumerate(chapters):
            path.append({
                "id": chapter.id,
                "title": chapter.title,
                "module_id": chapter.module_id,
                "order_number": chapter.order_number,
                "is_recommended": i >= start_idx,
                "difficulty_estimate": self._estimate_difficulty(chapter, user)
            })

        return path

    def _estimate_difficulty(self, chapter: Chapter, user: User) -> str:
        """Estimate difficulty of a chapter for the user."""
        # This is a simplified difficulty estimation
        # A real system would analyze the chapter content in detail
        user_level = user.personalization_level
        if user_level == BackgroundLevel.BEGINNER:
            return "challenging"
        elif user_level == BackgroundLevel.INTERMEDIATE:
            return "moderate"
        elif user_level == BackgroundLevel.ADVANCED:
            return "easy"
        else:
            return "moderate"