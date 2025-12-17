from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api import auth_routes, content_routes, chat_routes, translation_routes, personalization_routes
from .database import engine
from .models.user import Base

# Create tables in database
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(content_routes.router, prefix="/api/content", tags=["content"])
app.include_router(chat_routes.router, prefix="/api/chat", tags=["chat"])
app.include_router(translation_routes.router, prefix="/api/translate", tags=["translation"])
app.include_router(translation_router.translation_router, prefix="/api/translation", tags=["translation"])
app.include_router(personalization_routes.router, prefix="/api/personalization", tags=["personalization"])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.app_version}


# Root endpoint
@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.app_name} v{settings.app_version}"}