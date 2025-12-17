"""
Chat API Router for Physical AI & Humanoid Robotics Platform
"""
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime

from src.models.user import UserResponse
from src.models.chat_session import ChatMessage, ChatSession
from src.services.database import get_db_session
from src.services.auth_service import get_current_user
from src.services.chat_service import chat_service
from src.services.personalization_service import personalization_service
from src.services.translation_service import translation_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
chat_router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}}
)

@chat_router.post("/message")
async def send_message(
    message: str,
    session_id: Optional[str] = None,
    translate_to_urdu: bool = False,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Send a message to the chatbot
    """
    try:
        # Get user preferences for personalization
        user_preferences = await personalization_service.get_user_preferences(db_session, current_user.id)

        # Prepare context for the chat service
        context = {
            "user_id": current_user.id,
            "user_background": {
                "software_background": current_user.software_background,
                "hardware_background": current_user.hardware_background,
                "robotics_experience": current_user.robotics_experience
            },
            "preferences": user_preferences
        }

        # Process the message using the chat service
        response = await chat_service.process_chat_message(
            user_id=current_user.id,
            message=message,
            session_id=session_id,
            context=context
        )

        if not response.get("response"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process message"
            )

        # Translate response to Urdu if requested
        if translate_to_urdu:
            translation_result = await translation_service.translate_text(
                response["response"],
                target_lang="ur",
                source_lang="en"
            )
            if translation_result["success"]:
                response["urdu_response"] = translation_result["translated_text"]

        # Return the response
        return {
            "message": message,
            "response": response["response"],
            "urdu_response": response.get("urdu_response"),
            "sources": response.get("sources", []),
            "session_id": response["session_id"],
            "context_used": response.get("context_used", False),
            "timestamp": response["timestamp"],
            "user_id": current_user.id
        }

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@chat_router.get("/sessions")
async def get_chat_sessions(
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get all chat sessions for the user
    """
    try:
        # This would fetch chat sessions from the database
        # For now, we'll return a placeholder response
        # In a real implementation, this would query the database
        return {
            "user_id": current_user.id,
            "sessions": [
                {
                    "session_id": "session_1",
                    "title": "Introduction to ROS 2",
                    "last_message": "How do I create a publisher?",
                    "last_message_time": "2023-10-15T10:30:00Z",
                    "message_count": 12
                },
                {
                    "session_id": "session_2",
                    "title": "Navigation in Gazebo",
                    "last_message": "How to configure Nav2 for humanoid?",
                    "last_message_time": "2023-10-14T15:45:00Z",
                    "message_count": 8
                }
            ],
            "total_sessions": 2,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@chat_router.get("/session/{session_id}")
async def get_session_history(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get chat history for a specific session
    """
    try:
        # Get conversation history from chat service
        history = await chat_service.get_conversation_history(session_id)

        return {
            "session_id": session_id,
            "history": history,
            "message_count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@chat_router.post("/translate")
async def translate_message(
    text: str,
    target_lang: str = "ur",
    source_lang: str = "en",
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Translate text using the translation service
    """
    try:
        translation_result = await translation_service.translate_text(
            text=text,
            target_lang=target_lang,
            source_lang=source_lang
        )

        if not translation_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Translation failed"
            )

        # Save translation to database for future reference
        await translation_service.save_translation_to_db(
            db_session=db_session,
            original_text=translation_result["original_text"],
            translated_text=translation_result["translated_text"],
            source_lang=translation_result["source_language"],
            target_lang=translation_result["target_language"],
            content_type="chat_message",
            user_id=current_user.id
        )

        return translation_result

    except Exception as e:
        logger.error(f"Error translating message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@chat_router.post("/knowledge-base/add")
async def add_to_knowledge_base(
    content: str,
    document_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Add content to the knowledge base for RAG
    """
    try:
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required to add to knowledge base"
            )

        success = await chat_service.add_document_to_knowledge_base(
            content=content,
            document_id=document_id,
            metadata=metadata or {}
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add document to knowledge base"
            )

        return {
            "success": True,
            "document_id": document_id or "auto-generated",
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding to knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@chat_router.get("/relevant-documents")
async def get_relevant_documents(
    query: str,
    top_k: int = 5,
    db_session: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Get relevant documents for a query
    """
    try:
        documents = await chat_service.get_relevant_documents(
            query=query,
            top_k=top_k
        )

        return {
            "query": query,
            "documents": documents,
            "count": len(documents),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting relevant documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@chat_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Parse the message
            try:
                message_data = json.loads(data)
                message = message_data.get("message", "")
                session_id = message_data.get("session_id", f"session_{client_id}")
                translate_to_urdu = message_data.get("translate_to_urdu", False)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                continue

            # Process the message using the chat service
            response = await chat_service.process_chat_message(
                user_id=client_id,
                message=message,
                session_id=session_id
            )

            # Translate response to Urdu if requested
            if translate_to_urdu:
                translation_result = await translation_service.translate_text(
                    response["response"],
                    target_lang="ur",
                    source_lang="en"
                )
                if translation_result["success"]:
                    response["urdu_response"] = translation_result["translated_text"]

            # Prepare response
            response_data = {
                "type": "message_response",
                "original_message": message,
                "response": response["response"],
                "urdu_response": response.get("urdu_response"),
                "session_id": response["session_id"],
                "timestamp": response["timestamp"]
            }

            # Send response back to client
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket.close()


@chat_router.get("/health")
async def chat_health():
    """
    Chat service health check
    """
    return {
        "status": "healthy",
        "service": "chat",
        "timestamp": datetime.utcnow().isoformat()
    }


__all__ = ["chat_router"]