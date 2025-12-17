# Physical AI & Humanoid Robotics Backend

This backend implements the complete system for the Physical AI & Humanoid Robotics platform with authentication, personalization, RAG chatbot, and Claude Code Subagents.

## Architecture

The backend is built with FastAPI and includes:

- **Authentication**: Better-Auth integration with user profiles
- **Content Management**: Module/chapter management with personalization
- **RAG System**: Retrieval-Augmented Generation for chatbot functionality
- **Translation Service**: Urdu translation engine
- **Subagent System**: Claude Code Subagents for robotics tasks
- **Database**: Neon Postgres with SQLAlchemy ORM
- **Vector Store**: Qdrant for document embeddings

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (Neon)
- **ORM**: SQLAlchemy
- **Vector Store**: Qdrant
- **Authentication**: Better-Auth
- **Translation**: Custom translation engine
- **AI/ML**: OpenAI, Transformers
- **TypeScript**: For type safety

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:

```env
DATABASE_URL=postgresql://username:password@localhost/dbname
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_openai_api_key
BETTER_AUTH_SECRET=your_secret_key
NEON_DB_URL=your_neon_db_url
```

## Running the Server

```bash
uvicorn main:app --reload
```