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
            search_results = self.vector_store.similarity_search(question, k=top_k * 2)  # Get more results for persona filtering
            
            # Apply minimum relevance score filter
            min_score = guardrails.get('min_relevance_score', 0.0)
            filtered_results = [r for r in search_results if r['score'] >= min_score]
            
            # Apply persona-aware chunk selection and re-ranking
            persona = guardrails.get('persona', 'None')
            persona_threshold = guardrails.get('persona_threshold', 0.2)
            
            if persona != 'None':
                # Apply persona filtering and check if threshold is met
                persona_filtered_results, max_persona_score = self._filter_chunks_by_persona(filtered_results, persona, question)
                
                # Check if persona relevance meets threshold
                if max_persona_score >= persona_threshold:
                    filtered_results = persona_filtered_results
                    # Add metadata to indicate persona filtering was applied
                    for result in filtered_results:
                        result['metadata']['persona_filtering_applied'] = True
                        result['metadata']['persona_threshold_met'] = True
                else:
                    # Fall back to generic retrieval if persona relevance is too low
                    for result in filtered_results:
                        result['metadata']['persona_filtering_applied'] = False
                        result['metadata']['persona_threshold_met'] = False
                        result['metadata']['max_persona_score'] = max_persona_score
            
            # Limit to top_k results after persona filtering
            filtered_results = filtered_results[:top_k]
            
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
    
    def _filter_chunks_by_persona(self, chunks: List[Dict[str, Any]], persona: str, question: str) -> tuple[List[Dict[str, Any]], float]:
        """
        Filter and re-rank chunks based on persona relevance using Hybrid Retrieval Pattern
        
        Args:
            chunks: List of retrieved chunks
            persona: Target persona
            question: User question
            
        Returns:
            Tuple of (filtered and re-ranked chunks, maximum persona score)
        """
        persona_keywords = self._get_persona_keywords(persona)
        question_lower = question.lower()
        
        # Score each chunk based on persona relevance
        scored_chunks = []
        max_persona_score = 0
        
        for chunk in chunks:
            content_lower = chunk['content'].lower()
            metadata_lower = str(chunk.get('metadata', {})).lower()
            
            # Calculate persona relevance score
            persona_score = 0
            
            # Check for persona-specific keywords in content
            for keyword in persona_keywords:
                if keyword in content_lower:
                    persona_score += 2  # Higher weight for content matches
                if keyword in metadata_lower:
                    persona_score += 1  # Lower weight for metadata matches
            
            # Check for question-related keywords in content
            question_words = question_lower.split()
            for word in question_words:
                if len(word) > 3 and word in content_lower:  # Only consider meaningful words
                    persona_score += 0.5
            
            # Track maximum persona score
            max_persona_score = max(max_persona_score, persona_score)
            
            # Combine original similarity score with persona score
            combined_score = chunk['score'] + (persona_score * 0.1)  # Weight persona score appropriately
            
            scored_chunks.append({
                **chunk,
                'persona_score': persona_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score (original similarity + persona relevance)
        scored_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Update the original score with combined score for display
        for chunk in scored_chunks:
            chunk['score'] = chunk['combined_score']
            chunk['metadata']['persona_score'] = chunk['persona_score']
        
        return scored_chunks, max_persona_score
    
    def _get_persona_keywords(self, persona: str) -> List[str]:
        """
        Get persona-specific keywords for filtering
        
        Args:
            persona: Target persona
            
        Returns:
            List of relevant keywords
        """
        persona_keywords = {
            'CISO': [
                'security', 'cybersecurity', 'risk', 'threat', 'vulnerability', 'incident', 'breach',
                'access control', 'authentication', 'authorization', 'encryption', 'firewall',
                'security policy', 'security framework', 'compliance', 'audit', 'governance',
                'security awareness', 'training', 'monitoring', 'detection', 'response'
            ],
            'DPO': [
                'data protection', 'privacy', 'gdpr', 'consent', 'lawful basis', 'data subject',
                'data controller', 'data processor', 'dpo', 'data protection officer',
                'breach notification', 'right to be forgotten', 'data portability',
                'privacy by design', 'data minimization', 'purpose limitation',
                'storage limitation', 'accuracy', 'integrity', 'confidentiality'
            ],
            'Accountant': [
                'financial', 'cost', 'budget', 'expense', 'revenue', 'profit', 'loss',
                'accounting', 'bookkeeping', 'audit', 'financial reporting', 'tax',
                'compliance', 'regulatory', 'financial risk', 'investment', 'roi',
                'cost-benefit', 'financial impact', 'budget allocation', 'financial planning'
            ],
            'Layman': [
                'simple', 'easy', 'understand', 'explain', 'basic', 'fundamental',
                'practical', 'everyday', 'common', 'general', 'public', 'citizen',
                'rights', 'protection', 'privacy', 'personal', 'individual'
            ],
            'Student': [
                'learn', 'study', 'education', 'academic', 'research', 'theory',
                'concept', 'principle', 'example', 'case study', 'analysis',
                'understanding', 'knowledge', 'learning', 'curriculum', 'course'
            ],
            'CEO': [
                'business', 'strategy', 'leadership', 'management', 'executive',
                'corporate', 'organization', 'company', 'enterprise', 'stakeholder',
                'competitive advantage', 'market', 'growth', 'innovation', 'vision',
                'mission', 'goals', 'objectives', 'performance', 'success'
            ],
            'Financial Product Consumer': [
                'consumer', 'customer', 'client', 'user', 'individual', 'personal',
                'financial product', 'service', 'account', 'transaction', 'payment',
                'credit', 'loan', 'insurance', 'investment', 'savings', 'banking'
            ],
            'Legal Counsel': [
                'legal', 'law', 'regulation', 'statute', 'legislation', 'court',
                'litigation', 'liability', 'contract', 'agreement', 'compliance',
                'legal obligation', 'legal requirement', 'legal framework', 'jurisdiction',
                'precedent', 'case law', 'legal opinion', 'legal advice'
            ],
            'IT Manager': [
                'technical', 'technology', 'system', 'infrastructure', 'network',
                'software', 'hardware', 'database', 'server', 'cloud', 'api',
                'integration', 'implementation', 'deployment', 'maintenance',
                'support', 'troubleshooting', 'performance', 'scalability'
            ],
            'HR Manager': [
                'employee', 'staff', 'workforce', 'human resources', 'personnel',
                'recruitment', 'hiring', 'training', 'development', 'performance',
                'workplace', 'employment', 'labor', 'workplace policy', 'employee rights',
                'workplace culture', 'team', 'management', 'supervision'
            ],
            'Marketing Manager': [
                'marketing', 'advertising', 'promotion', 'campaign', 'brand',
                'customer', 'client', 'audience', 'target', 'market', 'sales',
                'revenue', 'growth', 'engagement', 'conversion', 'analytics',
                'digital marketing', 'social media', 'content', 'strategy'
            ],
            'Small Business Owner': [
                'small business', 'entrepreneur', 'startup', 'owner', 'founder',
                'business growth', 'scalability', 'resources', 'budget', 'cost',
                'practical', 'implementation', 'operations', 'management',
                'customer service', 'local business', 'community', 'flexibility'
            ],
            'Developer': [
                'code', 'programming', 'development', 'software', 'application',
                'api', 'database', 'backend', 'frontend', 'integration',
                'implementation', 'technical', 'architecture', 'framework',
                'library', 'tool', 'debugging', 'testing', 'deployment'
            ],
            'Auditor': [
                'audit', 'compliance', 'verification', 'evidence', 'documentation',
                'review', 'assessment', 'evaluation', 'control', 'procedure',
                'standard', 'requirement', 'regulatory', 'governance', 'risk',
                'internal audit', 'external audit', 'audit trail', 'findings'
            ]
        }
        
        return persona_keywords.get(persona, [])
