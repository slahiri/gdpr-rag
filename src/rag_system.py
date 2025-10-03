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
    
    def query_with_guardrails(self, question: str, guardrails: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system with guardrails applied
        
        Args:
            question: User question
            guardrails: Guardrails configuration
            top_k: Number of documents to retrieve
            
        Returns:
            Query result with answer and sources
        """
        try:
            # Apply domain restriction check
            if guardrails.get('domain_restriction') != 'None':
                if not self._is_domain_relevant(question, guardrails['domain_restriction']):
                    return {
                        "success": True,
                        "answer": f"I can only assist with {guardrails['domain_restriction']} related questions. Please ask a question related to {guardrails['domain_restriction']}.",
                        "sources": [],
                        "scores": [],
                        "guardrails_applied": True
                    }
            
            # Retrieve relevant documents
            search_results = self.vector_store.similarity_search(question, k=top_k)
            
            # Apply minimum relevance score filter
            min_score = guardrails.get('min_relevance_score', 0.0)
            filtered_results = [r for r in search_results if r['score'] >= min_score]
            
            # Check if sources are required
            if guardrails.get('require_sources', True) and not filtered_results:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Please ensure you have uploaded relevant documents or try rephrasing your question.",
                    "sources": [],
                    "scores": [],
                    "guardrails_applied": True
                }
            
            # Use filtered results or original results based on guardrails
            final_results = filtered_results if guardrails.get('require_sources', True) else search_results
            
            # Generate response using Bedrock with guardrails
            bedrock_response = self.bedrock_client.generate_response_with_guardrails(
                query=question,
                context_documents=final_results,
                guardrails=guardrails
            )
            
            # Extract sources and scores with full text content
            sources = []
            scores = []
            for result in final_results:
                sources.append({
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "score": result["score"],
                    "content_length": len(result["content"]),
                    "content_preview": result["content"][:150] + "..." if len(result["content"]) > 150 else result["content"]
                })
                scores.append(result["score"])
            
            return {
                "success": bedrock_response["success"],
                "answer": bedrock_response["response"],
                "sources": sources,
                "scores": scores,
                "model_used": bedrock_response["model_used"],
                "context_sources": bedrock_response["context_sources"],
                "guardrails_applied": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "scores": [],
                "guardrails_applied": True
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
            
            # Extract sources and scores with full text content
            sources = []
            scores = []
            for result in search_results:
                sources.append({
                    "content": result["content"],  # Show full text chunk
                    "metadata": result["metadata"],
                    "score": result["score"],
                    "content_length": len(result["content"]),  # Add content length for reference
                    "content_preview": result["content"][:150] + "..." if len(result["content"]) > 150 else result["content"]  # Preview for summaries
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
    
    def _is_domain_relevant(self, question: str, domain: str) -> bool:
        """
        Check if a question is relevant to the specified domain
        
        Args:
            question: User question
            domain: Domain to check against
            
        Returns:
            True if relevant, False otherwise
        """
        domain_keywords = {
            'GDPR': ['gdpr', 'data protection', 'privacy', 'personal data', 'data subject', 'consent', 'lawful basis', 'data controller', 'data processor', 'dpo', 'data protection officer', 'breach notification', 'right to be forgotten', 'data portability', 'privacy by design'],
            'Data Protection': ['data protection', 'personal data', 'privacy', 'data security', 'data breach', 'data retention', 'data minimization', 'purpose limitation'],
            'Privacy Law': ['privacy', 'privacy law', 'data protection', 'personal information', 'privacy rights', 'consent', 'opt-in', 'opt-out'],
            'Compliance': ['compliance', 'regulatory', 'audit', 'governance', 'policy', 'procedure', 'risk assessment', 'controls'],
            'Legal': ['legal', 'law', 'regulation', 'statute', 'legislation', 'court', 'litigation', 'liability', 'contract']
        }
        
        question_lower = question.lower()
        keywords = domain_keywords.get(domain, [])
        
        # Check if any domain keywords are present in the question
        return any(keyword in question_lower for keyword in keywords)
