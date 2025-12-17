"""
Qdrant Vector Store Service for Physical AI & Humanoid Robotics Platform
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional, Union, Any
import logging
import uuid
from contextlib import asynccontextmanager

from src.utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class QdrantService:
    """
    Service class for interacting with Qdrant vector database
    """
    def __init__(self):
        self.client = None
        self.collection_name = settings.COLLECTION_NAME

    async def initialize(self):
        """
        Initialize the Qdrant client and create collection if it doesn't exist
        """
        try:
            # Initialize client
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10.0
            )

            # Check if collection exists
            try:
                await self.client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists")
            except Exception:
                # Create collection if it doesn't exist
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection '{self.collection_name}'")

            logger.info("Qdrant service initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing Qdrant service: {e}")
            raise

    async def close(self):
        """
        Close the Qdrant client connection
        """
        if self.client:
            await self.client.aclose()

    async def add_document(
        self,
        document_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any] = None,
        module_id: Optional[int] = None,
        chapter_id: Optional[int] = None
    ) -> bool:
        """
        Add a document to the vector store
        """
        try:
            # Prepare payload
            payload = {
                "content": content,
                "module_id": module_id,
                "chapter_id": chapter_id,
                "created_at": __import__('datetime').datetime.utcnow().isoformat()
            }

            if metadata:
                payload.update(metadata)

            # Add point to collection
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=document_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            logger.debug(f"Added document {document_id} to Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error adding document to Qdrant: {e}")
            return False

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector store
        """
        try:
            # Prepare filters if provided
            qdrant_filters = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, (list, tuple)):
                        # Handle list of values (OR condition)
                        conditions = [
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=v)
                            )
                            for v in value
                        ]
                        filter_conditions.append(
                            models.Filter(
                                should=conditions
                            )
                        )
                    else:
                        # Handle single value
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

                if filter_conditions:
                    qdrant_filters = models.Filter(
                        must=filter_conditions
                    )

            # Perform search
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=qdrant_filters,
                score_threshold=min_score
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "module_id": result.payload.get("module_id"),
                    "chapter_id": result.payload.get("chapter_id"),
                    "score": result.score,
                    "payload": result.payload
                })

            logger.debug(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            return []

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID
        """
        try:
            records = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[document_id],
                with_payload=True,
                with_vectors=False
            )

            if records and len(records) > 0:
                record = records[0]
                return {
                    "id": record.id,
                    "content": record.payload.get("content", ""),
                    "module_id": record.payload.get("module_id"),
                    "chapter_id": record.payload.get("chapter_id"),
                    "payload": record.payload
                }

            return None

        except Exception as e:
            logger.error(f"Error retrieving document from Qdrant: {e}")
            return None

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store
        """
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[document_id]
            )

            logger.debug(f"Deleted document {document_id} from Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error deleting document from Qdrant: {e}")
            return False

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document in the vector store
        """
        try:
            # Get current document to merge metadata
            current_doc = await self.get_document(document_id)
            if not current_doc:
                logger.warning(f"Document {document_id} not found for update")
                return False

            # Prepare new payload
            new_payload = current_doc["payload"]
            if content:
                new_payload["content"] = content
            if metadata:
                new_payload.update(metadata)

            # Prepare point to update
            point = PointStruct(
                id=document_id,
                vector=embedding if embedding else current_doc.get("vector", []),
                payload=new_payload
            )

            # Update the point
            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            logger.debug(f"Updated document {document_id} in Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error updating document in Qdrant: {e}")
            return False

    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        """
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


# Global instance
qdrant_service = QdrantService()

async def init_qdrant():
    """
    Initialize Qdrant service
    """
    await qdrant_service.initialize()
    logger.info("Qdrant service initialized")


# Context manager for Qdrant operations
@asynccontextmanager
async def get_qdrant_service():
    """
    Context manager to get Qdrant service instance
    """
    yield qdrant_service


__all__ = [
    "QdrantService",
    "qdrant_service",
    "init_qdrant",
    "get_qdrant_service"
]