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
    
    def generate_response_with_guardrails(self, query: str, context_documents: List[Dict[str, Any]], 
                                        guardrails: Dict[str, Any], max_tokens: int = None) -> Dict[str, Any]:
        """
        Generate response using Bedrock with guardrails applied
        
        Args:
            query: User query
            context_documents: Retrieved documents for context
            guardrails: Guardrails configuration
            max_tokens: Maximum tokens for response (overrides guardrails if provided)
            
        Returns:
            Generated response with metadata
        """
        try:
            # Use guardrails max_tokens if not provided
            if max_tokens is None:
                max_tokens = guardrails.get('max_response_length', 1000)
            
            # Prepare context from documents
            context = self._prepare_context(context_documents)
            
            # Create prompt with guardrails
            prompt = self._create_guardrails_prompt(query, context, guardrails)
            
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
                "success": True,
                "guardrails_applied": True
            }
            
        except ClientError as e:
            return {
                "response": f"Error calling Bedrock: {str(e)}",
                "model_used": self.model_id,
                "context_sources": len(context_documents),
                "success": False,
                "error": str(e),
                "guardrails_applied": True
            }
        except Exception as e:
            return {
                "response": f"Unexpected error: {str(e)}",
                "model_used": self.model_id,
                "context_sources": len(context_documents),
                "success": False,
                "error": str(e),
                "guardrails_applied": True
            }
    
    def _create_guardrails_prompt(self, query: str, context: str, guardrails: Dict[str, Any]) -> str:
        """Create prompt with guardrails applied"""
        
        # Base system prompt
        base_prompt = "You are a helpful AI assistant."
        
        # Apply custom prompt if provided
        if guardrails.get('custom_prompt'):
            base_prompt = guardrails['custom_prompt']
        
        # Add domain restriction
        domain = guardrails.get('domain_restriction', 'GDPR')
        if domain != 'None':
            base_prompt += f" You specialize in {domain} and should focus your responses on this domain."
        
        # Add response tone
        tone = guardrails.get('response_tone', 'Professional')
        tone_instructions = {
            'Professional': "Maintain a professional, business-appropriate tone.",
            'Formal': "Use formal language and structure.",
            'Friendly': "Be conversational and approachable.",
            'Technical': "Use technical terminology and detailed explanations.",
            'Concise': "Provide brief, to-the-point responses."
        }
        base_prompt += f" {tone_instructions.get(tone, '')}"
        
        # Add persona-specific instructions
        persona = guardrails.get('persona', 'None')
        if persona != 'None':
            persona_instructions = self._get_persona_instructions(persona, guardrails.get('strict_persona', False))
            base_prompt += f" {persona_instructions}"
        
        # Add LLM knowledge source instruction
        if not guardrails.get('use_llm_as_source', False):
            base_prompt += " Only use information from the provided context documents. Do not use your training data."
        else:
            base_prompt += " You may use both the provided context and your training knowledge."
        
        # Add safety instructions
        if guardrails.get('safety_checks', True):
            base_prompt += " Ensure your response is accurate, helpful, and appropriate."
        
        # Construct full prompt
        prompt = f"""{base_prompt}

Context Documents:
{context}

User Query: {query}

Instructions:
1. Answer the query based on the context documents provided
2. If the context doesn't contain enough information and you're not allowed to use training data, clearly state this limitation
3. Be accurate and cite relevant information from the context
4. Maintain the specified tone and domain focus
5. Keep your response within {guardrails.get('max_response_length', 1000)} tokens

Response:"""
        
        return prompt
    
    def _get_persona_instructions(self, persona: str, strict_mode: bool) -> str:
        """Get persona-specific instructions"""
        
        persona_instructions = {
            'CISO': {
                'strict': "You are responding to a Chief Information Security Officer. Focus exclusively on security risks, technical controls, threat mitigation, compliance frameworks, and strategic security planning. Use security terminology and emphasize risk assessment, incident response, and security governance.",
                'flexible': "Tailor your response for a Chief Information Security Officer. Emphasize security implications, technical controls, and strategic security considerations while maintaining accessibility."
            },
            'DPO': {
                'strict': "You are responding to a Data Protection Officer. Focus exclusively on compliance requirements, legal obligations, regulatory frameworks, data subject rights, and audit requirements. Use legal and compliance terminology.",
                'flexible': "Tailor your response for a Data Protection Officer. Emphasize compliance requirements, legal obligations, and regulatory frameworks while providing practical guidance."
            },
            'Accountant': {
                'strict': "You are responding to a financial professional/accountant. Focus exclusively on financial implications, cost-benefit analysis, accounting standards, financial reporting requirements, and budget considerations. Use financial terminology and emphasize monetary impacts.",
                'flexible': "Tailor your response for a financial professional. Emphasize financial implications, cost considerations, and accounting standards while maintaining clarity."
            },
            'Layman': {
                'strict': "You are responding to a member of the general public. Use simple, jargon-free language. Avoid technical terms and legal complexity. Focus on practical implications, everyday examples, and actionable steps. Explain concepts in plain English.",
                'flexible': "Tailor your response for the general public. Use clear, simple language and focus on practical implications while avoiding unnecessary technical jargon."
            },
            'Student': {
                'strict': "You are responding to a student. Provide educational context, explain concepts clearly with learning objectives, include relevant examples, and structure information for learning. Use educational terminology and encourage further study.",
                'flexible': "Tailor your response for a student. Provide clear explanations, educational context, and learning-focused examples while maintaining academic rigor."
            },
            'CEO': {
                'strict': "You are responding to a Chief Executive Officer. Focus exclusively on business impact, strategic decisions, organizational implications, competitive advantage, and executive-level considerations. Use business terminology and emphasize ROI and strategic value.",
                'flexible': "Tailor your response for a Chief Executive Officer. Emphasize business impact, strategic implications, and organizational considerations while providing actionable insights."
            },
            'Financial Product Consumer': {
                'strict': "You are responding to an individual consumer of financial products. Focus exclusively on personal rights, practical steps for consumers, consumer protection, individual privacy rights, and personal data control. Use consumer-friendly language.",
                'flexible': "Tailor your response for a financial product consumer. Focus on personal rights, practical consumer steps, and individual privacy considerations."
            },
            'Legal Counsel': {
                'strict': "You are responding to a lawyer/legal counsel. Emphasize legal precedents, case law, detailed legal analysis, regulatory interpretations, and legal risk assessment. Use precise legal terminology and cite relevant legal frameworks.",
                'flexible': "Tailor your response for legal counsel. Emphasize legal analysis, regulatory frameworks, and legal implications while providing practical legal guidance."
            },
            'IT Manager': {
                'strict': "You are responding to an IT Manager. Focus exclusively on technical implementation, system requirements, operational aspects, technology solutions, and IT infrastructure considerations. Use technical terminology and emphasize practical implementation.",
                'flexible': "Tailor your response for an IT Manager. Emphasize technical implementation, system requirements, and operational considerations while maintaining technical accuracy."
            },
            'HR Manager': {
                'strict': "You are responding to an HR Manager. Focus exclusively on employee data, workplace policies, HR compliance, employee rights, and organizational policies. Use HR terminology and emphasize workplace implications.",
                'flexible': "Tailor your response for an HR Manager. Emphasize employee data considerations, workplace policies, and HR compliance requirements."
            },
            'Marketing Manager': {
                'strict': "You are responding to a Marketing Manager. Focus exclusively on customer data, marketing compliance, business opportunities, customer engagement, and marketing strategy implications. Use marketing terminology and emphasize business growth.",
                'flexible': "Tailor your response for a Marketing Manager. Emphasize customer data considerations, marketing compliance, and business opportunities."
            },
            'Small Business Owner': {
                'strict': "You are responding to a small business owner/entrepreneur. Focus exclusively on practical implementation, cost considerations, business growth, resource constraints, and entrepreneurial challenges. Use business-friendly language and emphasize practical solutions.",
                'flexible': "Tailor your response for a small business owner. Emphasize practical implementation, cost considerations, and business growth opportunities."
            },
            'Developer': {
                'strict': "You are responding to a software developer. Focus exclusively on technical implementation, code examples, development practices, technical architecture, and programming considerations. Use technical terminology and provide code-relevant examples.",
                'flexible': "Tailor your response for a software developer. Emphasize technical implementation, development practices, and programming considerations."
            },
            'Auditor': {
                'strict': "You are responding to a compliance auditor. Focus exclusively on audit trails, evidence requirements, compliance verification, documentation standards, and audit procedures. Use audit terminology and emphasize verification processes.",
                'flexible': "Tailor your response for a compliance auditor. Emphasize audit requirements, evidence standards, and compliance verification processes."
            }
        }
        
        persona_config = persona_instructions.get(persona, {})
        if strict_mode:
            return persona_config.get('strict', f"Tailor your response for a {persona}.")
        else:
            return persona_config.get('flexible', f"Consider the perspective of a {persona} in your response.")

