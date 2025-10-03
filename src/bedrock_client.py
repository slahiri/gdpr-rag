"""
AWS Bedrock Client Module for GDPR RAG System
Handles LLM interactions using AWS Bedrock
"""

import json
import boto3
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError


class BedrockClient:
    """Handles AWS Bedrock interactions for LLM responses"""
    
    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        Initialize Bedrock client
        
        Args:
            region_name: AWS region
            model_id: Bedrock model ID to use
        """
        self.region_name = region_name
        self.model_id = model_id
        
        # Initialize Bedrock client
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
    
    def generate_response(self, query: str, context_documents: List[Dict[str, Any]], 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate response using Bedrock
        
        Args:
            query: User query
            context_documents: Retrieved documents for context
            max_tokens: Maximum tokens for response
            
        Returns:
            Generated response with metadata
        """
        try:
            # Prepare context from documents
            context = self._prepare_context(context_documents)
            
            # Create prompt for GDPR-specific responses
            prompt = self._create_gdpr_prompt(query, context)
            
            # Prepare request body based on model
            if "claude" in self.model_id.lower():
                request_body = self._create_claude_request(prompt, max_tokens)
            else:
                raise ValueError(f"Unsupported model: {self.model_id}")
            
            # Invoke model
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            return {
                "response": self._extract_response_text(response_body),
                "model_used": self.model_id,
                "context_sources": len(context_documents),
                "success": True
            }
            
        except ClientError as e:
            return {
                "response": f"Error calling Bedrock: {str(e)}",
                "model_used": self.model_id,
                "context_sources": len(context_documents),
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "response": f"Unexpected error: {str(e)}",
                "model_used": self.model_id,
                "context_sources": len(context_documents),
                "success": False,
                "error": str(e)
            }
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0)
            
            context_part = f"Document {i} (Relevance Score: {score:.3f}):\n"
            context_part += f"Source: {metadata.get('filename', 'Unknown')}\n"
            context_part += f"Content: {content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_gdpr_prompt(self, query: str, context: str) -> str:
        """Create GDPR-specific prompt"""
        prompt = f"""You are a GDPR compliance expert assistant. Your role is to provide accurate, helpful information about GDPR (General Data Protection Regulation) based on the provided documents.

Context Documents:
{context}

User Query: {query}

Instructions:
1. Answer the query based primarily on the provided context documents
2. If the context doesn't contain enough information, clearly state this limitation
3. Focus on GDPR compliance, data protection, privacy rights, and regulatory requirements
4. Provide specific references to relevant sections or articles when possible
5. If discussing legal requirements, emphasize that this is informational and recommend consulting legal professionals for specific cases
6. Be clear, concise, and professional in your response

Response:"""
        
        return prompt
    
    def _create_claude_request(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """Create request body for Claude models"""
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for more consistent responses
            "top_p": 0.9
        }
    
    def _extract_response_text(self, response_body: Dict[str, Any]) -> str:
        """Extract response text from Bedrock response"""
        if "content" in response_body:
            # Claude response format
            content_blocks = response_body["content"]
            if content_blocks and len(content_blocks) > 0:
                return content_blocks[0].get("text", "")
        
        # Fallback for other formats
        return str(response_body)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Bedrock connection"""
        try:
            # Simple test query
            test_response = self.generate_response(
                query="What is GDPR?",
                context_documents=[{
                    "content": "GDPR is the General Data Protection Regulation.",
                    "metadata": {"source": "test"},
                    "score": 1.0
                }],
                max_tokens=100
            )
            
            return {
                "success": test_response["success"],
                "message": "Connection successful" if test_response["success"] else f"Connection failed: {test_response.get('error', 'Unknown error')}"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}"
            }

