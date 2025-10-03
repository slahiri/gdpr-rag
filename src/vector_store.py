"""
Vector Store Module for GDPR RAG System
Handles vector database operations for document storage and retrieval
"""

import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorStore:
    """Handles vector database operations for document storage and retrieval"""
    
    def __init__(self, url: str = "http://localhost:6333", api_key: str = None, collection_name: str = "gdpr_docs"):
        """
        Initialize Qdrant vector store
        
        Args:
            url: Qdrant server URL
            api_key: Qdrant API key (for cloud instances)
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Qdrant client
        if api_key:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(url=url)
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,  # Dimension for all-MiniLM-L6-v2
                    distance=Distance.COSINE
                )
            )
    
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to Qdrant vector store
        
        Args:
            documents: List of Document objects to add
        """
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare points for Qdrant
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    **doc.metadata
                }
            )
            points.append(point)
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search in Qdrant
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        # Format results
        search_results = []
        for result in results:
            search_result = {
                "content": result.payload["text"],
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            }
            search_results.append(search_result)
        
        return search_results
    
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection"""
        info = self.client.get_collection(self.collection_name)
        return {
            "type": "qdrant",
            "collection_name": self.collection_name,
            "count": info.points_count
        }
