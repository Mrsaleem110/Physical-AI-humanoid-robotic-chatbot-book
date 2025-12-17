# Physical AI & Humanoid Robotics Platform

A comprehensive educational platform for humanoid robotics with AI-powered chatbot assistance, personalization, and multi-language support.

## Features

- **4 Modules × 4 Chapters**: Complete curriculum on humanoid robotics
- **AI-Powered Chatbot**: RAG-based assistant with robotics knowledge
- **Personalization**: Adaptive learning based on user preferences
- **Multi-Language Support**: Including Urdu translation capabilities
- **Claude Code Subagents**: Specialized robotics subagents for navigation, manipulation, perception, control, and interaction
- **ROS 2 Integration**: Complete ROS 2 ecosystem integration
- **Isaac Sim & Isaac ROS**: Advanced simulation and perception capabilities
- **Modern Web Interface**: React frontend with FastAPI backend

## Prerequisites

- Python 3.9+
- Node.js 16+
- PostgreSQL 12+
- Qdrant vector database
- Docker (optional, for containerization)

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables in a `.env` file (see `.env.example`)

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Set up environment variables in `frontend/.env`

### Documentation Setup

1. Navigate to the docs directory:
```bash
cd docs
```

2. Install Docusaurus dependencies:
```bash
npm install
```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo

# Database Configuration
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/humanoid_robotics_db

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Authentication
JWT_SECRET_KEY=your_super_secret_jwt_key_here_make_it_long_and_random

# Frontend Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
```

## Running the Application

### LibreTranslate Service (Required for Translation Feature)

Before starting the main application, you need to set up the LibreTranslate service for the translation functionality:

```bash
# Option 1: Using Docker Compose (recommended)
docker-compose up -d

# Option 2: Using pip
pip install libretranslate
libretranslate --host 0.0.0.0 --port 5000

# Option 3: Using the setup script (Windows)
setup-libretranslate.bat
```

### Backend (FastAPI)

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (React)

1. Start the frontend development server:
```bash
cd frontend
npm start
```

### Documentation (Docusaurus)

1. Start the documentation server:
```bash
cd docs
npm start
```

## Translation Feature

The application includes a real-time translation feature that allows users to translate book content into 8 languages:
- English (en)
- Urdu (ur)
- Hindi (hi)
- Spanish (es)
- Japanese (ja)
- Chinese (zh)
- French (fr)
- German (de)

To use the translation feature:
1. Make sure the LibreTranslate service is running on http://localhost:5000
2. Navigate to any page in the application
3. Click the "Translate" button (🌐 icon) in the top-right corner
4. Select your preferred language from the dropdown
5. The page content will be translated in real-time

The translation service is configured in the `.env` file with the `REACT_APP_LIBRETRANSLATE_URL` variable.

## Database Setup

The application uses PostgreSQL with SQLAlchemy ORM. To set up the database:

1. Ensure PostgreSQL is running
2. Update the `DATABASE_URL` in your `.env` file
3. The application will automatically create tables on startup

## Vector Database Setup

The application uses Qdrant for vector storage:

1. Install Qdrant (https://qdrant.tech/documentation/quick-start/)
2. Start Qdrant service
3. Update the `QDRANT_URL` in your `.env` file

## API Documentation

The API documentation is automatically available at:
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Key Endpoints

- `GET /`: API root endpoint
- `GET /health`: Health check
- `POST /api/v1/chat/message`: Chatbot endpoint
- `POST /api/v1/translation/translate`: Translation endpoint
- `GET /api/v1/personalization/preferences`: User preferences
- `POST /api/v1/robotics/command`: Robotics command endpoint

## Claude Code Subagents

The platform includes specialized subagents for robotics tasks:

- **Navigation Subagent**: Path planning and movement
- **Manipulation Subagent**: Object grasping and manipulation
- **Perception Subagent**: Object detection and scene understanding
- **Control Subagent**: Joint control and robot state management
- **Interaction Subagent**: Speech and gesture capabilities

## Architecture

### Backend Structure
```
backend/
├── src/
│   ├── api/          # API routers and routes
│   ├── models/       # Database models
│   ├── services/     # Business logic services
│   ├── agents/       # Claude Code subagents
│   └── utils/        # Utility functions
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/   # React components
│   ├── services/     # API service calls
│   ├── types/        # TypeScript type definitions
│   └── styles/       # CSS styles
```

### Documentation Structure
```
docs/
├── docs/            # Documentation content
│   ├── module-1/    # ROS 2 fundamentals
│   ├── module-2/    # Physics simulation
│   ├── module-3/    # Advanced simulation
│   └── module-4/    # AI integration
```

## Configuration

### API Settings

The application is configured through environment variables. Key settings include:

- `APP_ENV`: Environment (development, production)
- `DEBUG`: Debug mode (true/false)
- `ALLOWED_ORIGINS`: CORS configuration
- `SIMILARITY_THRESHOLD`: RAG similarity threshold
- `CHAT_HISTORY_LIMIT`: Number of messages to keep in history

### Database Models

The application includes the following key models:

- **User**: User accounts and preferences
- **Chapter**: Educational content chapters
- **ChatSession**: Chat session management
- **Translation**: Translation caching
- **RoboticsTask**: Robotics task management
- **Personalization**: User personalization settings

## Deployment

### Production Deployment

For production deployment:

1. Set `APP_ENV=production` in your environment
2. Use a production-ready database
3. Set up a reverse proxy (nginx, Apache)
4. Use a process manager (PM2, systemd) for the backend
5. Implement proper logging and monitoring

### Docker Deployment

Docker configuration files can be added for containerized deployment:

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Security

- JWT-based authentication
- CORS protection
- Input validation and sanitization
- Secure session management
- Environment variable-based secrets management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For support, please open an issue in the repository or contact the development team.

## License

This project is licensed under the MIT License - see the LICENSE file for details.