"""
Chat Service for Physical AI & Humanoid Robotics Platform
Implements RAG (Retrieval-Augmented Generation) functionality
"""
from typing import List, Dict, Optional, Any
import logging
import asyncio
from datetime import datetime
import openai

from src.utils.config import settings
from src.services.robotics_service import robotics_service
from src.agents.agentic_sphere_agent import AgenticSphereAgent
from src.models.chat_session import ChatMessage

# Only import langchain components if we need RAG functionality
# For now, we'll handle agentic sphere functionality without full RAG dependencies

# Set up logging
logger = logging.getLogger(__name__)

class ChatService:
    """
    Service class for chat functionality with optional RAG
    """
    def __init__(self):
        # Initialize OpenAI
        openai.api_key = settings.OPENAI_API_KEY

        # Initialize vector store and chain as None initially
        self.vector_store = None
        self.chain = None
        self.embeddings = None
        self.llm = None

        # Conversation history
        self.conversation_history = {}

    async def initialize(self):
        """
        Initialize the chat service with optional RAG functionality
        """
        try:
            # Try to initialize RAG components, but make it optional
            try:
                # Import langchain components only when needed
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
                from langchain_community.vectorstores import Qdrant
                from langchain.chains import ConversationalRetrievalChain

                # Initialize embeddings and LLM
                self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
                self.llm = ChatOpenAI(
                    model_name=settings.OPENAI_MODEL,
                    temperature=0.7,
                    openai_api_key=settings.OPENAI_API_KEY
                )

                # Initialize vector store with Qdrant
                self.vector_store = Qdrant(
                    client=qdrant_service.client,
                    collection_name=settings.COLLECTION_NAME,
                    embeddings=self.embeddings
                )

                # Create conversational retrieval chain
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store.as_retriever(
                        search_kwargs={
                            "k": 5,  # Number of documents to retrieve
                            "score_threshold": settings.SIMILARITY_THRESHOLD
                        }
                    ),
                    return_source_documents=True,
                    verbose=True
                )

                logger.info("Chat service with RAG initialized successfully")
            except ImportError:
                # If langchain components are not available, continue with agentic sphere only
                logger.warning("RAG dependencies not available, initializing with agentic sphere functionality only")
                pass

            logger.info("Chat service basic initialization completed")

        except Exception as e:
            logger.error(f"Error initializing chat service: {e}")
            raise

    async def process_chat_message(
        self,
        user_id: int,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message using RAG or specialized agents
        """
        try:
            # Check if the message is about Agentic Sphere
            if self._is_agentic_sphere_query(message):
                # Route to Agentic Sphere agent
                response_text = await self._process_agentic_sphere_query(message)
                sources = []
            else:
                # Initialize conversation history for this user/session
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = []

                # Prepare chat history
                chat_history = self.conversation_history[session_id]

                # Create a custom prompt that incorporates context
                if context:
                    # Customize the prompt based on user context (background, preferences, etc.)
                    custom_context = self._build_contextual_prompt(context)
                    message = f"{custom_context}\n\nUser question: {message}"

                # Check if RAG functionality is available
                if self.chain is not None:
                    # Use the conversational chain to get response
                    result = self.chain({
                        "question": message,
                        "chat_history": chat_history
                    })

                    # Extract response and source documents
                    response_text = result.get("answer", "I'm sorry, I couldn't find a relevant answer.")
                    source_documents = result.get("source_documents", [])

                    # Format source information
                    sources = []
                    for doc in source_documents:
                        sources.append({
                            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            "metadata": doc.metadata
                        })
                else:
                    # RAG is not available, return a message indicating this
                    response_text = "I'm sorry, but the advanced knowledge base is not currently available. However, I can still help with Agentic Sphere related queries."
                    sources = []

                # Update conversation history
                self.conversation_history[session_id].append((message, response_text))

                # Limit conversation history to prevent memory issues
                if len(self.conversation_history[session_id]) > 10:
                    self.conversation_history[session_id] = self.conversation_history[session_id][-10:]

            response = {
                "response": response_text,
                "sources": sources,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "context_used": bool(context)
            }

            logger.debug(f"Processed chat message for user {user_id}")
            return response

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request.",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _is_agentic_sphere_query(self, message: str) -> bool:
        """
        Check if the message is about Agentic Sphere
        """
        message_lower = message.lower()

        # Check for exact phrase matches first (most important)
        exact_matches = [
            "agentic sphere"
        ]

        if any(exact_match in message_lower for exact_match in exact_matches):
            return True

        # Check for other related keywords
        agentic_sphere_keywords = [
            "agentic", "sphere", "muhammad saleem",
            "digital mind", "autonomous agent", "ai agent", "intelligent action",
            "vision into action", "business idea", "agent creation", "decision making",
            "turning vision into intelligent action", "agentic sphere agent"
        ]
        return any(keyword in message_lower for keyword in agentic_sphere_keywords)

    async def _process_agentic_sphere_query(self, message: str) -> str:
        """
        Process a query specifically for the Agentic Sphere agent
        """
        try:
            logger.info(f"Processing agentic sphere query: {message}")

            # Create a task for the Agentic Sphere agent
            task = {
                "type": "info_request",
                "description": message,
                "parameters": {}
            }

            # Create an instance of the Agentic Sphere agent
            agent = AgenticSphereAgent()
            logger.info(f"Created agent: {agent.name}")

            # Execute the task using the agent directly
            result = await agent.execute(task)
            logger.info(f"Agent execution result: success={result.get('success')}")

            if result.get("success"):
                # Extract the response from the agent's result
                agent_result = result.get("result", {})
                logger.info(f"Agent result keys: {list(agent_result.keys()) if agent_result else 'None'}")

                if "info" in agent_result:
                    info = agent_result["info"]
                    logger.info(f"Info keys: {list(info.keys()) if info else 'None'}")

                    # Build response with safe access to info fields
                    response_parts = []

                    if info.get('name'):
                        response_parts.append(f"**{info['name']}**")
                    if info.get('ceo'):
                        response_parts.append(f"CEO: {info['ceo']}")
                    if info.get('vision'):
                        response_parts.append(f"Vision: {info['vision']}")
                    if info.get('mission'):
                        response_parts.append(f"Mission: {info['mission']}")
                    if info.get('description'):
                        response_parts.append(f"Description: {info['description']}")
                    if info.get('capabilities'):
                        caps = info['capabilities'] if isinstance(info['capabilities'], list) else [info['capabilities']]
                        response_parts.append(f"Capabilities: {', '.join(caps)}")
                    if info.get('approach'):
                        response_parts.append(f"Approach: {info['approach']}")
                    if info.get('value_proposition'):
                        response_parts.append(f"Value Proposition: {info['value_proposition']}")
                    if info.get('tagline'):
                        response_parts.append(f"Tagline: {info['tagline']}")

                    if response_parts:
                        return "\n\n".join(response_parts)
                    else:
                        # Fallback if info structure is unexpected
                        return f"**Agentic Sphere**: {message} - I have information about this topic."
                else:
                    # If no specific info, return the general result message
                    message_result = agent_result.get("message", "I have information about Agentic Sphere.")
                    logger.info(f"Returning agent message: {message_result}")
                    return message_result or "I have information about Agentic Sphere."
            else:
                # If agent execution fails, return a default response
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Agent execution failed: {error_msg}")
                return """**Agentic Sphere** is a futuristic AI platform where bold business ideas are transformed into intelligent, autonomous AI agents that plan, decide, and execute with precision. We don't just build tools—we create digital minds that work for your business 24/7, scaling operations, automating decisions, and unlocking new growth.

Led by **Muhammad Saleem**, CEO and an **AI-native visionary who thinks like an artificial intelligent agent**, Agentic Sphere stands at the intersection of innovation and execution. His vision is to redefine how businesses operate by turning imagination into living, thinking AI systems.

**Agentic Sphere — Turning Vision into Intelligent Action. 🚀**"""

        except Exception as e:
            logger.error(f"Error processing Agentic Sphere query: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return """**Agentic Sphere** is a futuristic AI platform where bold business ideas are transformed into intelligent, autonomous AI agents that plan, decide, and execute with precision. We don't just build tools—we create digital minds that work for your business 24/7, scaling operations, automating decisions, and unlocking new growth.

Led by **Muhammad Saleem**, CEO and an **AI-native visionary who thinks like an artificial intelligent agent**, Agentic Sphere stands at the intersection of innovation and execution. His vision is to redefine how businesses operate by turning imagination into living, thinking AI systems.

**Agentic Sphere — Turning Vision into Intelligent Action. 🚀**"""

    def _build_contextual_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build a contextual prompt based on user context
        """
        background_info = []

        # Add user background information
        if context.get("software_background"):
            background_info.append(f"User has {context['software_background']} software background.")

        if context.get("hardware_background"):
            background_info.append(f"User has {context['hardware_background']} hardware background.")

        if context.get("robotics_experience"):
            background_info.append(f"User has {context['robotics_experience']} robotics experience.")

        # Add content preferences
        if context.get("preferred_learning_style"):
            background_info.append(f"User prefers {context['preferred_learning_style']} learning style.")

        if context.get("difficulty_preference"):
            background_info.append(f"User prefers {context['difficulty_preference']} difficulty level.")

        # Combine context information
        if background_info:
            return f"Context: {' '.join(background_info)} Provide explanations appropriate for this user."
        else:
            return "Provide clear and comprehensive explanations."

    async def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        """
        history = self.conversation_history.get(session_id, [])
        formatted_history = []

        for i, (user_msg, ai_resp) in enumerate(history):
            formatted_history.append({
                "turn": i + 1,
                "user_message": user_msg,
                "ai_response": ai_resp,
                "timestamp": datetime.utcnow().isoformat()  # In real implementation, store actual timestamps
            })

        return formatted_history

    async def clear_conversation_history(self, session_id: str):
        """
        Clear conversation history for a session
        """
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    async def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant documents for a query without processing through the full chain
        """
        try:
            # Perform similarity search
            results = await qdrant_service.search_similar(
                query_embedding=self._get_embedding(query),
                limit=top_k,
                min_score=settings.SIMILARITY_THRESHOLD
            )

            documents = []
            for result in results:
                documents.append({
                    "content": result["content"],
                    "module_id": result["module_id"],
                    "chapter_id": result["chapter_id"],
                    "score": result["score"],
                    "metadata": result["payload"]
                })

            return documents

        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return []

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI
        """
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"  # Using a reliable embedding model
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return [0.0] * 1536  # Return zeros if embedding fails

    async def add_document_to_knowledge_base(
        self,
        content: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a document to the knowledge base for RAG
        """
        try:
            if not document_id:
                document_id = f"doc_{int(datetime.utcnow().timestamp())}_{hash(content) % 10000}"

            # Get embedding for the content
            embedding = self._get_embedding(content)

            # Add to Qdrant
            success = await qdrant_service.add_document(
                document_id=document_id,
                content=content,
                embedding=embedding,
                metadata=metadata or {}
            )

            if success:
                logger.info(f"Added document {document_id} to knowledge base")
            else:
                logger.error(f"Failed to add document {document_id} to knowledge base")

            return success

        except Exception as e:
            logger.error(f"Error adding document to knowledge base: {e}")
            return False

    async def remove_document_from_knowledge_base(self, document_id: str) -> bool:
        """
        Remove a document from the knowledge base
        """
        try:
            success = await qdrant_service.delete_document(document_id)
            if success:
                logger.info(f"Removed document {document_id} from knowledge base")
            else:
                logger.error(f"Failed to remove document {document_id} from knowledge base")

            return success

        except Exception as e:
            logger.error(f"Error removing document from knowledge base: {e}")
            return False

    async def update_document_in_knowledge_base(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document in the knowledge base
        """
        try:
            success = await qdrant_service.update_document(
                document_id=document_id,
                content=content,
                metadata=metadata
            )

            if success:
                logger.info(f"Updated document {document_id} in knowledge base")
            else:
                logger.error(f"Failed to update document {document_id} in knowledge base")

            return success

        except Exception as e:
            logger.error(f"Error updating document in knowledge base: {e}")
            return False


# Global instance
chat_service = ChatService()

async def init_chat_service():
    """
    Initialize the chat service
    """
    await chat_service.initialize()
    logger.info("Chat service initialized")


__all__ = [
    "ChatService",
    "chat_service",
    "init_chat_service"
]