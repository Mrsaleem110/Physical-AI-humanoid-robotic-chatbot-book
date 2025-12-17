

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from ..database import get_db
from ..models.user import User
from ..models.chat_session import ChatSession, ChatMessage
from ..services.rag_service import RAGService
from ..services.auth_service import AuthService

router = APIRouter()
security = HTTPBearer()
rag_service = RAGService()
auth_service = AuthService()


@router.post("/sessions", response_model=dict)
async def create_chat_session(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """Create a new chat session."""
    user_id = None
    if credentials:
        token = credentials.credentials
        user = auth_service.get_current_user(token, db)
        if user:
            user_id = user.id

    session = rag_service.create_chat_session(db, user_id)
    return {
        "id": session.id,
        "user_id": session.user_id,
        "created_at": session.created_at,
        "is_active": session.is_active
    }


@router.get("/sessions/{session_id}", response_model=dict)
async def get_chat_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific chat session."""
    session = rag_service.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    return {
        "id": session.id,
        "user_id": session.user_id,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "is_active": session.is_active,
        "context_metadata": session.context_metadata
    }


@router.get("/sessions/{session_id}/messages", response_model=List[dict])
async def get_chat_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """Get messages from a chat session."""
    messages = rag_service.get_chat_messages(db, session_id, limit, offset)
    return [
        {
            "id": msg.id,
            "session_id": msg.session_id,
            "sender_type": msg.sender_type,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "message_type": msg.message_type,
            "sources": msg.sources
        }
        for msg in messages
    ]


@router.post("/sessions/{session_id}/messages", response_model=dict)
async def send_chat_message(
    session_id: str,
    content: str,
    context_type: Optional[str] = Query("full_book", description="Type of context: full_book, selected_text, chapter_specific"),
    selected_content: Optional[str] = Query(None, description="Specific content to limit responses to (for selected_text context)"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """Send a message to the chat session and receive AI response."""
    # Verify session exists
    session = rag_service.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Add user message to session
    user_message = rag_service.add_chat_message(
        db, session_id, "user", content, "question"
    )

    # Process the query with RAG service
    response_data = rag_service.process_query(content, session_id, selected_content)

    # Add AI response to session
    ai_message = rag_service.add_chat_message(
        db, session_id, "ai", response_data["response"], "answer", response_data["sources"]
    )

    return {
        "user_message": {
            "id": user_message.id,
            "content": user_message.content,
            "timestamp": user_message.timestamp
        },
        "ai_response": {
            "id": ai_message.id,
            "content": ai_message.content,
            "timestamp": ai_message.timestamp,
            "sources": ai_message.sources
        },
        "query": response_data["query"],
        "sources": response_data["sources"]
    }


@router.post("/sessions/{session_id}/messages/context", response_model=dict)
async def set_chat_context(
    session_id: str,
    context_type: str = Query(..., description="Type of context: full_book, selected_text, chapter_specific"),
    selected_content_id: Optional[str] = Query(None, description="ID of specific content to limit responses to"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """Set the context for a chat session (e.g., 'answer only from selected text')."""
    session = rag_service.get_chat_session(db, session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Update session context
    session.context_metadata = {
        "context_type": context_type,
        "selected_content_id": selected_content_id
    }

    db.commit()
    db.refresh(session)

    return {
        "session_id": session.id,
        "context_type": context_type,
        "selected_content_id": selected_content_id,
        "updated_at": session.updated_at
    }