"""
RAG System Module for GDPR RAG System
Main orchestrator that combines all components
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .pdf_processor import PDFProcessor
from .vector_store import VectorStore
from .bedrock_client import BedrockClient


class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG system
        
        Args:
            config: Configuration dictionary
        """
        # Load environment variables
        load_dotenv()
        
        # Set default configuration
        self.config = config or {}
        
        # Initialize components
        self.pdf_processor = PDFProcessor(
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 200)
        )
        
        # Initialize Qdrant vector store
        self.vector_store = VectorStore(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=self.config.get("collection_name", "gdpr_docs")
        )
        
        # Initialize Bedrock client
        self.bedrock_client = BedrockClient(
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        )
    
    def upload_document(self, pdf_file, filename: str = None) -> Dict[str, Any]:
        """
        Upload and process a PDF document
        
        Args:
            pdf_file: Uploaded PDF file object
            filename: Name of the PDF file
            
        Returns:
            Processing result with metadata
        """
        try:
            # Process PDF
            documents = self.pdf_processor.process_pdf(pdf_file, filename)
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"Successfully processed {len(documents)} chunks from {filename or 'PDF'}",
                "chunks_created": len(documents),
                "filename": filename
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing document: {str(e)}",
                "chunks_created": 0,
                "filename": filename
            }
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Query result with answer and sources
        """
        try:
            # Retrieve relevant documents
            search_results = self.vector_store.similarity_search(question, k=top_k)
            
            if not search_results:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "scores": []
                }
            
            # Generate response using Bedrock
            bedrock_response = self.bedrock_client.generate_response(
                query=question,
                context_documents=search_results
            )
            
            # Extract sources and scores
            sources = []
            scores = []
            for result in search_results:
                sources.append({
                    "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "metadata": result["metadata"],
                    "score": result["score"]
                })
                scores.append(result["score"])
            
            return {
                "success": bedrock_response["success"],
                "answer": bedrock_response["response"],
                "sources": sources,
                "scores": scores,
                "model_used": bedrock_response["model_used"],
                "context_sources": bedrock_response["context_sources"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "scores": []
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and information"""
        try:
            # Test vector store
            vector_info = self.vector_store.get_collection_info()
            
            # Test Bedrock connection
            bedrock_status = self.bedrock_client.test_connection()
            
            return {
                "vector_store": vector_info,
                "bedrock": bedrock_status,
                "system_ready": bedrock_status["success"]
            }
            
        except Exception as e:
            return {
                "vector_store": {"error": str(e)},
                "bedrock": {"success": False, "message": str(e)},
                "system_ready": False
            }
    
    def clear_documents(self) -> Dict[str, Any]:
        """Clear all documents from the Qdrant collection"""
        try:
            # For Qdrant, delete the collection and recreate it
            self.vector_store.client.delete_collection(self.vector_store.collection_name)
            from qdrant_client.models import VectorParams, Distance
            self.vector_store.client.create_collection(
                collection_name=self.vector_store.collection_name,
                vectors_config=VectorParams(
                    size=384,  # Dimension for all-MiniLM-L6-v2
                    distance=Distance.COSINE
                )
            )
            
            return {
                "success": True,
                "message": "All documents cleared successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error clearing documents: {str(e)}"
            }
