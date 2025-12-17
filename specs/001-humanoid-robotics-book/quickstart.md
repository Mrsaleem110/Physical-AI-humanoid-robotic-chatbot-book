# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Getting Started

This guide will help you set up and run the Physical AI & Humanoid Robotics Book system locally.

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (for containerized services)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Set up backend services**:
   ```bash
   # Navigate to backend directory
   cd backend

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up frontend**:
   ```bash
   # Navigate to frontend directory
   cd frontend

   # Install dependencies
   npm install
   ```

4. **Set up documentation**:
   ```bash
   # Navigate to docs directory
   cd docs

   # Install dependencies
   npm install
   ```

### Environment Configuration

Create `.env` files for each service:

**Backend (.env)**:
```env
DATABASE_URL="postgresql://user:password@localhost:5432/humanoid_book"
QDRANT_URL="http://localhost:6333"
QDRANT_API_KEY="your-qdrant-api-key"
OPENAI_API_KEY="your-openai-api-key"
BETTER_AUTH_SECRET="your-auth-secret"
BETTER_AUTH_URL="http://localhost:3000"
```

**Frontend (.env)**:
```env
REACT_APP_API_URL="http://localhost:8000"
REACT_APP_DOCS_URL="http://localhost:3001"
```

### Running Services

1. **Start database and vector store**:
   ```bash
   docker-compose up -d
   ```

2. **Run backend**:
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn src.main:app --reload --port 8000
   ```

3. **Run frontend**:
   ```bash
   cd frontend
   npm start
   ```

4. **Run documentation site**:
   ```bash
   cd docs
   npm start
   ```

### API Contracts

The system exposes several API endpoints:

- Authentication: `POST /api/auth/login`, `POST /api/auth/register`
- Content: `GET /api/content/chapters`, `GET /api/content/chapter/{id}`
- Chat: `POST /api/chat/{session_id}/message`, `GET /api/chat/{session_id}`
- Translation: `POST /api/translate`, `GET /api/translate/languages`
- Personalization: `POST /api/personalization/level`, `GET /api/personalization/profile`

### Initial Setup

1. **Run database migrations**:
   ```bash
   cd backend
   python -m src.database.migrate
   ```

2. **Initialize content**:
   ```bash
   python -m src.content.init
   ```

3. **Index content for search**:
   ```bash
   python -m src.search.index
   ```

### Testing

Run tests for each component:

**Backend tests**:
```bash
cd backend
python -m pytest tests/
```

**Frontend tests**:
```bash
cd frontend
npm test
```

**E2E tests**:
```bash
npm run test:e2e
```

### Development Workflow

1. **Content Development**: Add/modify chapters in the `docs/docs/` directory following the module structure
2. **API Development**: Add new endpoints in `backend/src/api/` with corresponding services in `backend/src/services/`
3. **Frontend Development**: Create new components in `frontend/src/components/` and pages in `frontend/src/pages/`
4. **Subagent Development**: Add new agents in `backend/src/agents/` following the standardized interface

### Common Tasks

- **Update chapter content**: Modify files in `docs/docs/` and restart the docs server
- **Add new user**: Use the registration form or API endpoint
- **Test chat functionality**: Use the embedded chat interface or API directly
- **Change personalization level**: Update user profile settings
- **Translate content**: Use the translation interface or API endpoint

### Troubleshooting

- **Database connection issues**: Verify PostgreSQL is running and credentials are correct
- **API not responding**: Check that backend is running on port 8000
- **Docs not building**: Verify Node.js version and dependencies
- **Translation errors**: Check API keys and translation service availability
- **Chat not working**: Verify OpenAI API key and Qdrant connection