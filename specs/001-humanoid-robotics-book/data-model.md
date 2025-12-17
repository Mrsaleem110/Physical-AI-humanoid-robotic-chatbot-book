# Data Model: Physical AI & Humanoid Robotics Book

## Entity Definitions

### User
**Description**: Represents a book reader with profile data including software/hardware/robotics experience levels, authentication credentials, and personalization preferences.

**Fields**:
- id (UUID, primary key)
- email (string, unique, required)
- password_hash (string, required)
- first_name (string, optional)
- last_name (string, optional)
- created_at (timestamp, required)
- updated_at (timestamp, required)
- software_background (enum: beginner, intermediate, advanced, none, required)
- hardware_background (enum: beginner, intermediate, advanced, none, required)
- robotics_experience (enum: beginner, intermediate, advanced, none, required)
- personalization_level (enum: beginner, intermediate, advanced, default: beginner)
- preferred_language (string, default: "en")

**Validation Rules**:
- Email must be valid email format
- Password must meet security requirements
- Background fields must be one of the allowed enum values
- Personalization level defaults to beginner if not specified

**State Transitions**:
- Created → Active (after email verification)
- Active → Inactive (if account deactivated)

### Chapter
**Description**: Represents a book chapter with content that can be personalized and translated, containing text, code blocks, diagrams, and interactive elements.

**Fields**:
- id (UUID, primary key)
- module_id (UUID, foreign key to Module)
- title (string, required)
- slug (string, unique, required, auto-generated from title)
- content_en (text, required)
- content_ur (text, optional)
- order_number (integer, required)
- created_at (timestamp, required)
- updated_at (timestamp, required)
- is_published (boolean, default: true)

**Validation Rules**:
- Title must be 1-200 characters
- Slug must be unique across all chapters
- Order number must be positive
- Content must be provided in at least one language

### Module
**Description**: Represents a collection of 4 chapters covering a specific robotics topic (ROS 2, Simulation, AI-Brain, VLA).

**Fields**:
- id (UUID, primary key)
- title (string, required)
- slug (string, unique, required, auto-generated from title)
- description (text, optional)
- order_number (integer, required)
- created_at (timestamp, required)
- updated_at (timestamp, required)
- is_published (boolean, default: true)

**Validation Rules**:
- Title must be 1-200 characters
- Slug must be unique across all modules
- Order number must be positive

### ChatSession
**Description**: Represents an AI chatbot conversation with history, context, and user-specific information.

**Fields**:
- id (UUID, primary key)
- user_id (UUID, foreign key to User, nullable for anonymous sessions)
- created_at (timestamp, required)
- updated_at (timestamp, required)
- is_active (boolean, default: true)
- context_metadata (JSON, optional)

**Validation Rules**:
- User ID can be null for anonymous sessions
- Session becomes inactive after 24 hours of inactivity

### ChatMessage
**Description**: Represents a single message in a chat session, either from user or AI.

**Fields**:
- id (UUID, primary key)
- session_id (UUID, foreign key to ChatSession)
- sender_type (enum: user, ai, required)
- content (text, required)
- timestamp (timestamp, required)
- message_type (enum: question, answer, system, default: question)

**Validation Rules**:
- Sender type must be one of the allowed values
- Content must be provided

### Translation
**Description**: Represents a translated version of content with preserved formatting and language metadata.

**Fields**:
- id (UUID, primary key)
- content_id (string, required, references chapter/section ID)
- content_type (enum: chapter, section, paragraph, required)
- source_language (string, default: "en")
- target_language (string, required)
- original_content (text, required)
- translated_content (text, required)
- translation_quality_score (float, 0-1, optional)
- created_at (timestamp, required)
- updated_at (timestamp, required)

**Validation Rules**:
- Source and target languages must be different
- Translation quality score must be between 0 and 1 if provided
- Content ID must reference an existing content item

### UserInteraction
**Description**: Represents user interactions with the book content, used for personalization and analytics.

**Fields**:
- id (UUID, primary key)
- user_id (UUID, foreign key to User)
- content_id (string, required, references chapter/section ID)
- content_type (enum: chapter, section, exercise, required)
- interaction_type (enum: view, personalize, translate, complete, required)
- interaction_data (JSON, optional)
- timestamp (timestamp, required)

**Validation Rules**:
- User ID must reference an existing user
- Content ID must be provided
- Interaction type must be one of the allowed values

### RoboticsWorkflow
**Description**: Represents an executable robotics task combining ROS 2, simulation, and AI planning components.

**Fields**:
- id (UUID, primary key)
- name (string, required)
- description (text, optional)
- workflow_definition (JSON, required)
- subagent_config (JSON, required)
- created_at (timestamp, required)
- updated_at (timestamp, required)
- is_active (boolean, default: true)

**Validation Rules**:
- Name must be 1-200 characters
- Workflow definition must be valid JSON
- Subagent config must be valid JSON