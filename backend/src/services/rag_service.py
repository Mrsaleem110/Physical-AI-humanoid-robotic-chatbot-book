from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from ..config import settings
from ..models.chat_session import ChatSession, ChatMessage
from sqlalchemy.orm import Session
import uuid
import logging

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        # Collection name for book content
        self.collection_name = "book_content"

        # Initialize the Qdrant collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """Initialize the Qdrant collection for book content."""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def add_content_to_index(self, content_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add content to the vector index for retrieval."""
        # In a real implementation, we would use an embedding model to create vectors
        # For this example, we'll simulate the process
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode(content).tolist()

        # Upsert the content to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=content_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "content_id": content_id,
                        **(metadata or {})
                    }
                )
            ]
        )

    def search_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant content based on the query."""
        # In a real implementation, we would use an embedding model to create query vector
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query).tolist()

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        # Extract relevant content from search results
        results = []
        for result in search_results:
            results.append({
                "content_id": result.payload.get("content_id"),
                "content": result.payload.get("content"),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k not in ["content", "content_id"]}
            })

        return results

    def generate_response(self, query: str, context: List[Dict[str, Any]], selected_text_only: bool = False) -> str:
        """Generate a response using OpenAI based on the query and context."""
        if selected_text_only and context:
            # Use only the provided context (selected text mode)
            context_text = "\n\n".join([item["content"] for item in context])
        elif context:
            # Use retrieved context from the RAG system
            context_text = "\n\n".join([item["content"] for item in context])
        else:
            # No context available
            context_text = "No relevant content found in the book."

        # Prepare the prompt for OpenAI
        prompt = f"""
        You are an AI assistant for the Physical AI & Humanoid Robotics book.
        Answer the user's question based on the provided context from the book.

        Context: {context_text}

        User's question: {query}

        Please provide a helpful answer based on the context. If the context doesn't contain
        relevant information, politely let the user know that the information isn't available
        in the current book content.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for a robotics book. Answer questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating a response. Please try again."

    def process_query(self, query: str, session_id: str = None, selected_content: str = None) -> Dict[str, Any]:
        """Process a user query and return a response with sources."""
        # Search for relevant content
        if selected_content:
            # Use only the selected content if provided
            context = [{
                "content_id": "selected_content",
                "content": selected_content,
                "score": 1.0,
                "metadata": {}
            }]
            search_results = context
        else:
            # Search in the entire book
            search_results = self.search_content(query, limit=5)

        # Generate response
        response = self.generate_response(
            query,
            search_results,
            selected_text_only=bool(selected_content)
        )

        # Prepare sources
        sources = []
        for result in search_results[:3]:  # Limit to top 3 sources
            sources.append({
                "content_id": result["content_id"],
                "score": result["score"],
                "metadata": result["metadata"]
            })

        return {
            "response": response,
            "sources": sources,
            "query": query
        }

    def create_chat_session(self, db: Session, user_id: Optional[str] = None) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())

        db_session = ChatSession(
            id=session_id,
            user_id=user_id,
            context_metadata={"selected_text_only": False}  # Default context mode
        )

        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        return db_session

    def add_chat_message(self, db: Session, session_id: str, sender_type: str, content: str,
                        message_type: str = "question", sources: List[Dict[str, Any]] = None) -> ChatMessage:
        """Add a message to a chat session."""
        message_id = str(uuid.uuid4())

        db_message = ChatMessage(
            id=message_id,
            session_id=session_id,
            sender_type=sender_type,
            content=content,
            message_type=message_type,
            sources=sources or []
        )

        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        return db_message

    def get_chat_session(self, db: Session, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def get_chat_messages(self, db: Session, session_id: str, limit: int = 50, offset: int = 0) -> List[ChatMessage]:
        """Get messages from a chat session."""
        return db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.timestamp.desc()).offset(offset).limit(limit).all()