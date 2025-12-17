from typing import Any, Dict, List
from .base_agent import BaseAgent, AgentType, AgentSkill
from ..models.user import BackgroundLevel
import asyncio


class PersonalizationAgent(BaseAgent):
    """
    Personalization Subagent for content adaptation and user experience customization
    """

    def __init__(self):
        super().__init__(
            agent_type=AgentType.PERSONALIZATION,
            name="Personalization Agent",
            description="Specialized in content adaptation, user experience customization, and learning path optimization"
        )
        # Add relevant skills
        self.add_skill(AgentSkill.DATA_ANALYSIS)
        self.add_skill(AgentSkill.REASONING)
        self.add_skill(AgentSkill.ADAPTATION)
        self.add_skill(AgentSkill.LEARNING)

        # Initialize personalization components
        self.user_profiles = {}
        self.content_difficulty_mapping = {}
        self.adaptation_strategies = [
            "content_level_adaptation",
            "interface_customization",
            "learning_path_optimization",
            "recommendation_engine"
        ]

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute personalization tasks
        """
        task_type = task.get("type", "adapt_content")
        parameters = task.get("parameters", {})

        if task_type == "adapt_content":
            return await self._adapt_content(parameters)
        elif task_type == "customize_interface":
            return await self._customize_interface(parameters)
        elif task_type == "optimize_learning_path":
            return await self._optimize_learning_path(parameters)
        elif task_type == "generate_recommendations":
            return await self._generate_recommendations(parameters)
        elif task_type == "analyze_user_progress":
            return await self._analyze_user_progress(parameters)
        elif task_type == "update_user_profile":
            return await self._update_user_profile(parameters)
        else:
            return await self._perform_general_personalization(parameters)

    def can_handle(self, task: Dict[str, Any]) -> bool:
        """
        Determine if this agent can handle the given task
        """
        task_type = task.get("type", "")
        required_skills = task.get("required_skills", [])

        # Check if task type matches agent capabilities
        personalization_related = any(keyword in task_type.lower() for keyword in
                                    ["adapt", "customize", "personalize", "recommend", "profile", "learning", "path"])

        # Check if required skills are supported
        required_skills_supported = all(
            skill in [s.value for s in self.skills] for skill in required_skills
        )

        return personalization_related or required_skills_supported

    async def _adapt_content(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt content based on user profile and preferences
        """
        content = parameters.get("content", "")
        user_profile = parameters.get("user_profile", {})
        target_level = parameters.get("target_level", "adaptive")
        content_type = parameters.get("content_type", "text")

        # Determine the appropriate adaptation level
        if target_level == "adaptive":
            user_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)
        else:
            user_level = target_level

        # Adapt content based on user level
        adapted_content = self._apply_content_adaptation(content, user_level, content_type)

        return {
            "original_content_length": len(content),
            "adapted_content": adapted_content,
            "user_level": user_level,
            "content_type": content_type,
            "adaptation_applied": True,
            "status": "adapted"
        }

    def _apply_content_adaptation(self, content: str, user_level: str, content_type: str) -> str:
        """
        Apply adaptation rules based on user level
        """
        if user_level == BackgroundLevel.BEGINNER.value:
            # Add more explanations, examples, and context for beginners
            adapted_content = f"""
# Beginner-Friendly Content
## Key Concepts Explained
{content}

### Additional Explanations:
- **Key Term 1**: Simple definition and example
- **Key Term 2**: Simple definition and example

### Step-by-Step Guide:
1. Understand the basic concept
2. Look at the example
3. Try the exercise

### Practice Exercise:
Try applying this concept with the provided example.
"""
        elif user_level == BackgroundLevel.INTERMEDIATE.value:
            # Standard content for intermediate users
            adapted_content = f"""
# Intermediate Content
{content}

### Key Points:
- Important concept 1
- Important concept 2

### Application:
Apply these concepts in practice scenarios.
"""
        elif user_level == BackgroundLevel.ADVANCED.value:
            # More concise content for advanced users
            adapted_content = f"""
# Advanced Content
{content}

### Advanced Concepts:
- Complex concept 1
- Complex concept 2

### Implementation Notes:
Advanced implementation considerations.
"""
        else:
            # Default to beginner level if unknown
            adapted_content = f"""
# Beginner-Friendly Content
## Key Concepts Explained
{content}

### Additional Explanations:
- **Key Term 1**: Simple definition and example
- **Key Term 2**: Simple definition and example
"""

        return adapted_content

    async def _customize_interface(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Customize interface based on user preferences
        """
        user_profile = parameters.get("user_profile", {})
        interface_elements = parameters.get("interface_elements", [])
        customization_preferences = parameters.get("preferences", {})

        # Determine customization based on user profile
        interface_customization = {
            "theme": self._determine_theme(user_profile),
            "layout": self._determine_layout(user_profile),
            "navigation": self._determine_navigation(user_profile),
            "accessibility": self._determine_accessibility_features(user_profile),
            "custom_elements": self._customize_elements(interface_elements, user_profile)
        }

        return {
            "interface_customization": interface_customization,
            "user_profile": user_profile,
            "status": "customized"
        }

    def _determine_theme(self, user_profile: Dict[str, Any]) -> str:
        """
        Determine appropriate theme based on user profile
        """
        experience_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)

        if experience_level == BackgroundLevel.ADVANCED.value:
            return "dark"
        elif experience_level == BackgroundLevel.INTERMEDIATE.value:
            return "light"
        else:
            return "light"  # Beginners often prefer lighter themes

    def _determine_layout(self, user_profile: Dict[str, Any]) -> str:
        """
        Determine appropriate layout based on user profile
        """
        experience_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)

        if experience_level == BackgroundLevel.ADVANCED.value:
            return "dense"  # Advanced users can handle more information
        else:
            return "spacious"  # Beginners need more breathing room

    def _determine_navigation(self, user_profile: Dict[str, Any]) -> str:
        """
        Determine appropriate navigation based on user profile
        """
        experience_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)

        if experience_level == BackgroundLevel.ADVANCED.value:
            return "minimal"  # Advanced users prefer less clutter
        else:
            return "comprehensive"  # Beginners need clear navigation cues

    def _determine_accessibility_features(self, user_profile: Dict[str, Any]) -> List[str]:
        """
        Determine appropriate accessibility features based on user profile
        """
        features = []

        # Add features based on various factors
        if user_profile.get("visual_impairment", False):
            features.append("high_contrast")
            features.append("larger_text")

        if user_profile.get("motor_impairment", False):
            features.append("keyboard_navigation")

        # Beginners might benefit from additional guidance
        experience_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)
        if experience_level == BackgroundLevel.BEGINNER.value:
            features.append("tooltips")
            features.append("guided_tours")

        return features

    def _customize_elements(self, elements: List[str], user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Customize specific interface elements based on user profile
        """
        customized_elements = []

        for element in elements:
            customization = {
                "element": element,
                "visibility": True,  # Default to visible
                "position": "default",
                "size": "default"
            }

            # Customize based on experience level
            experience_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)

            if element == "help_button" and experience_level != BackgroundLevel.BEGINNER.value:
                customization["visibility"] = False  # Hide help for advanced users
            elif element == "advanced_features" and experience_level != BackgroundLevel.ADVANCED.value:
                customization["visibility"] = False  # Hide advanced features for beginners

            customized_elements.append(customization)

        return customized_elements

    async def _optimize_learning_path(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize learning path based on user profile and progress
        """
        user_profile = parameters.get("user_profile", {})
        current_progress = parameters.get("current_progress", {})
        learning_objectives = parameters.get("learning_objectives", [])
        available_content = parameters.get("available_content", [])

        # Calculate personalized learning path
        learning_path = self._calculate_learning_path(
            user_profile, current_progress, learning_objectives, available_content
        )

        return {
            "learning_path": learning_path,
            "user_profile": user_profile,
            "current_progress": current_progress,
            "total_steps": len(learning_path),
            "estimated_completion_time": self._estimate_completion_time(learning_path),
            "status": "optimized"
        }

    def _calculate_learning_path(self, user_profile: Dict[str, Any], current_progress: Dict[str, Any],
                                learning_objectives: List[str], available_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate a personalized learning path
        """
        # Determine starting point based on user profile
        experience_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)

        # Filter content based on user level and objectives
        filtered_content = []
        for content in available_content:
            content_level = content.get("difficulty_level", BackgroundLevel.INTERMEDIATE.value)

            # Include content based on user level
            if experience_level == BackgroundLevel.BEGINNER.value:
                if content_level in [BackgroundLevel.BEGINNER.value, BackgroundLevel.INTERMEDIATE.value]:
                    filtered_content.append(content)
            elif experience_level == BackgroundLevel.INTERMEDIATE.value:
                if content_level in [BackgroundLevel.BEGINNER.value, BackgroundLevel.INTERMEDIATE.value, BackgroundLevel.ADVANCED.value]:
                    filtered_content.append(content)
            else:  # Advanced
                filtered_content.append(content)

        # Sort content based on prerequisites and learning objectives
        sorted_content = self._sort_content_by_prerequisites(filtered_content, learning_objectives)

        # Create learning path with appropriate spacing
        learning_path = []
        for i, content in enumerate(sorted_content):
            learning_path.append({
                "id": content.get("id", f"content_{i}"),
                "title": content.get("title", f"Content {i+1}"),
                "type": content.get("type", "chapter"),
                "difficulty": content.get("difficulty_level", BackgroundLevel.INTERMEDIATE.value),
                "estimated_duration": content.get("estimated_duration", 30),  # minutes
                "prerequisites": content.get("prerequisites", []),
                "position": i + 1
            })

        return learning_path

    def _sort_content_by_prerequisites(self, content_list: List[Dict[str, Any]], objectives: List[str]) -> List[Dict[str, Any]]:
        """
        Sort content based on prerequisites and learning objectives
        """
        # This is a simplified sorting algorithm
        # A real implementation would use topological sorting
        sorted_content = []
        remaining_content = content_list.copy()

        # First, add content without prerequisites
        no_prereq_content = [c for c in remaining_content if not c.get("prerequisites")]
        sorted_content.extend(no_prereq_content)
        remaining_content = [c for c in remaining_content if c.get("prerequisites")]

        # Then try to add content whose prerequisites are satisfied
        max_iterations = len(remaining_content) * 2  # Prevent infinite loops
        iteration = 0

        while remaining_content and iteration < max_iterations:
            added_in_iteration = False

            for content in remaining_content[:]:  # Copy to avoid modification during iteration
                prereqs = content.get("prerequisites", [])
                satisfied_prereqs = [c for c in sorted_content if c.get("id") in prereqs]

                if len(satisfied_prereqs) == len(prereqs):
                    sorted_content.append(content)
                    remaining_content.remove(content)
                    added_in_iteration = True

            if not added_in_iteration:
                # If nothing was added, we have a circular dependency or unmet prerequisites
                break

            iteration += 1

        return sorted_content

    def _estimate_completion_time(self, learning_path: List[Dict[str, Any]]) -> float:
        """
        Estimate total completion time for the learning path
        """
        total_minutes = sum(item.get("estimated_duration", 30) for item in learning_path)
        return total_minutes

    async def _generate_recommendations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate personalized recommendations
        """
        user_profile = parameters.get("user_profile", {})
        current_content = parameters.get("current_content", "")
        interaction_history = parameters.get("interaction_history", [])
        available_content = parameters.get("available_content", [])

        # Generate recommendations based on user profile and history
        recommendations = self._calculate_recommendations(
            user_profile, current_content, interaction_history, available_content
        )

        return {
            "recommendations": recommendations,
            "user_profile": user_profile,
            "total_recommendations": len(recommendations),
            "status": "generated"
        }

    def _calculate_recommendations(self, user_profile: Dict[str, Any], current_content: str,
                                  interaction_history: List[Dict[str, Any]], available_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate personalized recommendations
        """
        # Calculate relevance scores for available content
        scored_content = []

        for content in available_content:
            relevance_score = self._calculate_relevance_score(
                content, user_profile, current_content, interaction_history
            )

            scored_content.append({
                "content": content,
                "relevance_score": relevance_score
            })

        # Sort by relevance and return top recommendations
        sorted_content = sorted(scored_content, key=lambda x: x["relevance_score"], reverse=True)

        recommendations = []
        for item in sorted_content[:5]:  # Top 5 recommendations
            content = item["content"]
            recommendations.append({
                "id": content.get("id"),
                "title": content.get("title"),
                "description": content.get("description", ""),
                "relevance_score": item["relevance_score"],
                "category": content.get("category", "general"),
                "difficulty": content.get("difficulty_level", BackgroundLevel.INTERMEDIATE.value)
            })

        return recommendations

    def _calculate_relevance_score(self, content: Dict[str, Any], user_profile: Dict[str, Any],
                                  current_content: str, interaction_history: List[Dict[str, Any]]) -> float:
        """
        Calculate relevance score for content based on user profile and history
        """
        score = 0.0

        # Factor 1: Content difficulty matching user level
        content_level = content.get("difficulty_level", BackgroundLevel.INTERMEDIATE.value)
        user_level = user_profile.get("background_level", BackgroundLevel.BEGINNER.value)

        if content_level == user_level:
            score += 0.3
        elif (content_level == BackgroundLevel.INTERMEDIATE.value and
              user_level in [BackgroundLevel.BEGINNER.value, BackgroundLevel.ADVANCED.value]):
            score += 0.2
        else:
            score += 0.1  # Some relevance even if difficulty doesn't match perfectly

        # Factor 2: Topic relevance to current content
        content_topics = set(content.get("topics", []))
        current_topics = set(current_content.split())  # Simplified
        topic_overlap = len(content_topics.intersection(current_topics))
        score += min(topic_overlap * 0.1, 0.3)  # Cap at 0.3

        # Factor 3: User interaction history
        if interaction_history:
            # Check if user has engaged with similar content before
            similar_content_interactions = [
                interaction for interaction in interaction_history
                if content.get("category") in str(interaction.get("content", ""))
            ]
            engagement_score = len(similar_content_interactions) * 0.05
            score += min(engagement_score, 0.2)  # Cap at 0.2

        # Factor 4: Content popularity/rating (if available)
        rating = content.get("rating", 3.0)  # Out of 5
        popularity_score = (rating / 5.0) * 0.2
        score += popularity_score

        return min(score, 1.0)  # Cap at 1.0

    async def _analyze_user_progress(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user progress and learning patterns
        """
        user_profile = parameters.get("user_profile", {})
        interaction_history = parameters.get("interaction_history", [])
        assessment_results = parameters.get("assessment_results", [])

        # Analyze progress patterns
        progress_analysis = {
            "learning_velocity": self._calculate_learning_velocity(interaction_history),
            "topic_mastery": self._analyze_topic_mastery(interaction_history),
            "engagement_patterns": self._analyze_engagement(interaction_history),
            "strengths": self._identify_strengths(interaction_history, assessment_results),
            "improvement_areas": self._identify_improvement_areas(interaction_history, assessment_results),
            "predicted_difficulty": self._predict_difficulty(interaction_history)
        }

        return {
            "progress_analysis": progress_analysis,
            "user_profile": user_profile,
            "status": "analyzed"
        }

    def _calculate_learning_velocity(self, interaction_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate learning velocity based on interaction history
        """
        if not interaction_history:
            return {"velocity": 0.0, "units": "topics_per_hour"}

        # Calculate based on time and completed content
        total_topics = len(interaction_history)
        # Simplified calculation - in a real system, you'd calculate based on time
        velocity = total_topics / 10.0  # Just a placeholder

        return {"velocity": velocity, "units": "topics_per_hour"}

    def _analyze_topic_mastery(self, interaction_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze mastery of different topics
        """
        topic_mastery = {}

        for interaction in interaction_history:
            topic = interaction.get("topic", "general")
            if topic not in topic_mastery:
                topic_mastery[topic] = {"count": 0, "mastery_score": 0.0}

            topic_mastery[topic]["count"] += 1
            # Simplified mastery calculation
            topic_mastery[topic]["mastery_score"] = min(
                topic_mastery[topic]["mastery_score"] + 0.1, 1.0
            )

        # Convert to simple scores
        return {topic: data["mastery_score"] for topic, data in topic_mastery.items()}

    def _analyze_engagement(self, interaction_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze user engagement patterns
        """
        if not interaction_history:
            return {"engagement_score": 0.0, "patterns": []}

        # Calculate engagement metrics
        total_interactions = len(interaction_history)
        interaction_types = {}

        for interaction in interaction_history:
            interaction_type = interaction.get("type", "general")
            interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1

        engagement_score = min(total_interactions * 0.1, 1.0)  # Cap at 1.0

        return {
            "engagement_score": engagement_score,
            "total_interactions": total_interactions,
            "interaction_types": interaction_types,
            "patterns": ["frequent_evening_use", "consistent_daily_sessions"]  # Example patterns
        }

    async def _update_user_profile(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user profile based on interactions and assessments
        """
        user_id = parameters.get("user_id", "")
        new_data = parameters.get("data", {})
        interaction_history = parameters.get("interaction_history", [])

        # Update profile in memory (in a real system, this would update the database)
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}

        # Merge new data with existing profile
        self.user_profiles[user_id].update(new_data)

        # Potentially update background level based on performance
        if interaction_history:
            updated_level = self._infer_user_level_from_interactions(interaction_history)
            if updated_level:
                self.user_profiles[user_id]["background_level"] = updated_level

        return {
            "user_id": user_id,
            "updated_profile": self.user_profiles[user_id],
            "status": "updated"
        }

    def _infer_user_level_from_interactions(self, interaction_history: List[Dict[str, Any]]) -> Optional[str]:
        """
        Infer user level from interaction patterns
        """
        if not interaction_history:
            return None

        # Simplified level inference based on interaction complexity
        # In a real system, you'd analyze assessment results, time spent, success rates, etc.
        recent_interactions = interaction_history[-5:]  # Look at last 5 interactions

        # Count advanced interactions
        advanced_count = sum(1 for interaction in recent_interactions
                           if interaction.get("complexity") == "advanced")

        if advanced_count >= 3:
            return BackgroundLevel.ADVANCED.value
        elif advanced_count >= 1:
            return BackgroundLevel.INTERMEDIATE.value
        else:
            return BackgroundLevel.BEGINNER.value

    async def _perform_general_personalization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a general personalization task
        """
        return {
            "parameters": parameters,
            "status": "processed",
            "message": "General personalization task completed"
        }