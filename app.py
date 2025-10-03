"""
GDPR RAG System - Streamlit Application
Main application for uploading PDFs and querying GDPR documents
"""

import streamlit as st
import os
import sys
from typing import Dict, Any
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_system import RAGSystem


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []


def setup_rag_system() -> RAGSystem:
    """Setup and initialize RAG system with Qdrant"""
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "collection_name": "gdpr_docs"
    }
    
    return RAGSystem(config)


def display_system_status(rag_system: RAGSystem):
    """Display system status"""
    with st.expander("System Status", expanded=False):
        status = rag_system.get_system_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vector Store")
            if "error" in status["vector_store"]:
                st.error(f"Error: {status['vector_store']['error']}")
            else:
                st.success(f"✅ {status['vector_store']['type'].title()} Connected")
                st.info(f"Documents: {status['vector_store'].get('count', status['vector_store'].get('total_vector_count', 0))}")
        
        with col2:
            st.subheader("AWS Bedrock")
            if status["bedrock"]["success"]:
                st.success("✅ Bedrock Connected")
            else:
                st.error(f"❌ Bedrock Error: {status['bedrock']['message']}")
        
        if status["system_ready"]:
            st.success("🚀 System Ready")
        else:
            st.warning("⚠️ System Not Ready - Check configuration")


def upload_documents_section(rag_system: RAGSystem):
    """Handle document upload section"""
    st.header("📄 Upload GDPR Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload GDPR-related PDF documents for analysis"
    )
    
    if uploaded_files:
        st.subheader("Uploaded Files")
        
        for uploaded_file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"📄 {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            with col2:
                if st.button("Process", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = rag_system.upload_document(uploaded_file, uploaded_file.name)
                        
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                            st.session_state.uploaded_files.append({
                                "name": uploaded_file.name,
                                "chunks": result["chunks_created"]
                            })
                        else:
                            st.error(f"❌ {result['message']}")
            
            with col3:
                if st.button("Clear All", key=f"clear_{uploaded_file.name}"):
                    clear_result = rag_system.clear_documents()
                    if clear_result["success"]:
                        st.success("🗑️ All documents cleared")
                        st.session_state.uploaded_files = []
                        st.rerun()
                    else:
                        st.error(f"❌ {clear_result['message']}")
    
    # Display uploaded files summary
    if st.session_state.uploaded_files:
        st.subheader("📊 Document Summary")
        df = pd.DataFrame(st.session_state.uploaded_files)
        st.dataframe(df, use_container_width=True)


def query_section(rag_system: RAGSystem):
    """Handle query section"""
    st.header("🔍 Query GDPR Documents")
    
    # Query input
    query = st.text_area(
        "Ask a question about GDPR compliance:",
        placeholder="e.g., What are the key requirements for data processing under GDPR?",
        height=100
    )
    
    # Query options
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of relevant documents to retrieve:", 1, 10, 5)
    with col2:
        if st.button("🔍 Search", type="primary"):
            if query.strip():
                with st.spinner("Searching documents and generating response..."):
                    # Use guardrails if configured
                    if 'guardrails' in st.session_state and st.session_state.guardrails.get('custom_prompt'):
                        result = rag_system.query_with_guardrails(
                            query, 
                            st.session_state.guardrails, 
                            top_k=top_k
                        )
                    else:
                        result = rag_system.query(query, top_k=top_k)
                    
                    # Store query in history
                    st.session_state.query_history.append({
                        "query": query,
                        "result": result,
                        "timestamp": pd.Timestamp.now()
                    })
                    
                    # Display results
                    display_query_results(result)
            else:
                st.warning("Please enter a question")


def display_query_results(result: Dict[str, Any]):
    """Display query results with enhanced source text display"""
    if not result["success"]:
        st.error(f"❌ Error: {result['answer']}")
        return
    
    # Main answer
    st.subheader("💡 Answer")
    st.markdown(result["answer"])
    
    # Sources and scores
    if result["sources"]:
        st.subheader("📚 Retrieved Text Chunks")
        st.markdown("*These are the actual text chunks from your documents that were used to generate the answer above.*")
        
                # Summary of retrieved chunks
                total_chunks = len(result["sources"])
                total_chars = sum(source.get("content_length", len(source["content"])) for source in result["sources"])
                avg_score = sum(result["scores"]) / len(result["scores"]) if result["scores"] else 0
                
                # Check if persona filtering was applied
                persona_applied = any(source.get("metadata", {}).get("persona_score") is not None for source in result["sources"])
                persona_threshold_met = any(source.get("metadata", {}).get("persona_threshold_met", False) for source in result["sources"])
                
                if persona_applied:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("📄 Chunks Retrieved", total_chunks)
                    with col2:
                        st.metric("📝 Total Characters", f"{total_chars:,}")
                    with col3:
                        st.metric("📊 Avg Relevance", f"{avg_score:.3f}")
                    with col4:
                        st.metric("📏 Avg Chunk Size", f"{total_chars//total_chunks if total_chunks > 0 else 0:,}")
                    with col5:
                        if persona_threshold_met:
                            st.metric("🎯 Persona Filtered", "✅ Yes")
                        else:
                            st.metric("🎯 Persona Filtered", "❌ Below Threshold")
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📄 Chunks Retrieved", total_chunks)
                    with col2:
                        st.metric("📝 Total Characters", f"{total_chars:,}")
                    with col3:
                        st.metric("📊 Avg Relevance", f"{avg_score:.3f}")
                    with col4:
                        st.metric("📏 Avg Chunk Size", f"{total_chars//total_chunks if total_chunks > 0 else 0:,}")
        
        st.divider()
        
        for i, source in enumerate(result["sources"], 1):
            # Create a more prominent display for each source
            col1, col2 = st.columns([1, 4])
            
            with col1:
                # Score badge
                score_color = "green" if source['score'] > 0.8 else "orange" if source['score'] > 0.6 else "red"
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border: 1px solid {score_color}; border-radius: 5px; margin: 5px 0;">
                    <strong>Source {i}</strong><br>
                    <span style="color: {score_color}; font-size: 18px; font-weight: bold;">
                        {source['score']:.3f}
                    </span><br>
                    <small>Relevance</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Text content in a highlighted box
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 4px solid #1f77b4; margin: 5px 0;">
                    <strong>📄 Text Chunk:</strong><br>
                    <div style="margin-top: 10px; line-height: 1.6;">
                        {source["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Metadata in a collapsible section
                with st.expander(f"📋 Metadata for Source {i}", expanded=False):
                    metadata = source["metadata"]
                    if metadata:
                        # Display metadata in a nice format
                        for key, value in metadata.items():
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                    else:
                        st.write("No metadata available")
                    
                        # Add chunk statistics
                        st.write("**Chunk Statistics:**")
                        st.write(f"**Content Length:** {source.get('content_length', len(source['content'])):,} characters")
                        st.write(f"**Relevance Score:** {source['score']:.3f}")
                        
                        # Show persona score if available
                        persona_score = source.get('metadata', {}).get('persona_score')
                        if persona_score is not None:
                            st.write(f"**Persona Score:** {persona_score:.1f}")
                            st.write(f"**Combined Score:** {source['score']:.3f} (semantic + persona)")
                            
                            # Show threshold status
                            threshold_met = source.get('metadata', {}).get('persona_threshold_met', False)
                            if threshold_met:
                                st.write("**Threshold Status:** ✅ Met")
                            else:
                                max_score = source.get('metadata', {}).get('max_persona_score', 0)
                                st.write(f"**Threshold Status:** ❌ Below (max: {max_score:.1f})")
                        
                        # Show content preview in metadata
                        if source.get('content_preview'):
                            st.write("**Content Preview:**")
                            st.code(source['content_preview'], language=None)
        
        # Score visualization
        if len(result["scores"]) > 1:
            st.subheader("📊 Relevance Score Distribution")
            scores_df = pd.DataFrame({
                "Source": [f"Source {i+1}" for i in range(len(result["scores"]))],
                "Score": result["scores"]
            })
            st.bar_chart(scores_df.set_index("Source"))
            
            # Add score interpretation
            avg_score = sum(result["scores"]) / len(result["scores"])
            if avg_score > 0.8:
                st.success(f"🎯 Excellent relevance! Average score: {avg_score:.3f}")
            elif avg_score > 0.6:
                st.info(f"✅ Good relevance. Average score: {avg_score:.3f}")
            else:
                st.warning(f"⚠️ Low relevance. Average score: {avg_score:.3f}")
    
    # Query metadata
    with st.expander("🔍 Query Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Used", result.get("model_used", "Unknown"))
        with col2:
            st.metric("Sources Found", result.get("context_sources", 0))
        with col3:
            avg_score = sum(result["scores"]) / len(result["scores"]) if result["scores"] else 0
            st.metric("Avg Relevance", f"{avg_score:.3f}")


def guardrails_section(rag_system):
    """Guardrails configuration section"""
    st.header("🛡️ Guardrails Configuration")
    st.markdown("Configure safety and behavior controls for the RAG system.")
    
    # Initialize guardrails in session state if not exists
    if 'guardrails' not in st.session_state:
        st.session_state.guardrails = {
            'use_llm_as_source': False,
            'custom_prompt': '',
            'domain_restriction': 'GDPR',
            'max_response_length': 1000,
            'require_sources': True,
            'min_relevance_score': 0.3,
            'safety_checks': True,
            'response_tone': 'Professional',
            'persona': 'None',
            'strict_persona': False,
            'persona_threshold': 0.2
        }
    
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Core Settings")
        
        # LLM as source option
        st.session_state.guardrails['use_llm_as_source'] = st.checkbox(
            "🤖 Use LLM as Knowledge Source",
            value=st.session_state.guardrails['use_llm_as_source'],
            help="Allow the LLM to use its training data in addition to uploaded documents"
        )
        
        # Domain restriction
        st.session_state.guardrails['domain_restriction'] = st.selectbox(
            "🎯 Domain Restriction",
            options=['GDPR', 'Data Protection', 'Privacy Law', 'Compliance', 'Legal', 'None'],
            index=['GDPR', 'Data Protection', 'Privacy Law', 'Compliance', 'Legal', 'None'].index(
                st.session_state.guardrails['domain_restriction']
            ),
            help="Restrict responses to specific domains"
        )
        
        # Response tone
        st.session_state.guardrails['response_tone'] = st.selectbox(
            "🎭 Response Tone",
            options=['Professional', 'Formal', 'Friendly', 'Technical', 'Concise'],
            index=['Professional', 'Formal', 'Friendly', 'Technical', 'Concise'].index(
                st.session_state.guardrails['response_tone']
            ),
            help="Set the tone for AI responses"
        )
        
        # Persona selection
        st.subheader("👤 User Persona")
        st.session_state.guardrails['persona'] = st.selectbox(
            "🎯 Target Persona",
            options=['None', 'CISO', 'DPO', 'Accountant', 'Layman', 'Student', 'CEO', 'Financial Product Consumer', 'Legal Counsel', 'IT Manager', 'HR Manager', 'Marketing Manager', 'Small Business Owner', 'Developer', 'Auditor'],
            index=['None', 'CISO', 'DPO', 'Accountant', 'Layman', 'Student', 'CEO', 'Financial Product Consumer', 'Legal Counsel', 'IT Manager', 'HR Manager', 'Marketing Manager', 'Small Business Owner', 'Developer', 'Auditor'].index(
                st.session_state.guardrails['persona']
            ),
            help="Select the persona/role to tailor responses for"
        )
        
                # Strict persona adherence
                st.session_state.guardrails['strict_persona'] = st.checkbox(
                    "🔒 Strictly Adhere to Persona",
                    value=st.session_state.guardrails['strict_persona'],
                    help="Enforce persona-specific language, examples, and context. If disabled, persona is used as guidance only."
                )
                
                # Persona threshold
                st.session_state.guardrails['persona_threshold'] = st.slider(
                    "🎯 Persona Relevance Threshold",
                    min_value=0.0,
                    max_value=2.0,
                    value=st.session_state.guardrails.get('persona_threshold', 0.2),
                    step=0.1,
                    help="Minimum persona relevance score required to apply persona filtering. If below threshold, uses generic retrieval."
                )
                
                # Show threshold explanation
                if st.session_state.guardrails['persona'] != 'None':
                    st.info(f"""
                    **Persona Threshold Logic:**
                    - **Above {st.session_state.guardrails['persona_threshold']}**: Apply persona-specific filtering and re-ranking
                    - **Below {st.session_state.guardrails['persona_threshold']}**: Fall back to generic semantic search
                    - **Score Calculation**: Content matches (+2), metadata matches (+1), question relevance (+0.5)
                    """)
        
        # Show persona description
        if st.session_state.guardrails['persona'] != 'None':
            persona_descriptions = {
                'CISO': 'Chief Information Security Officer - Focus on security risks, technical controls, and strategic security planning',
                'DPO': 'Data Protection Officer - Emphasize compliance requirements, legal obligations, and regulatory frameworks',
                'Accountant': 'Financial professional - Focus on financial implications, cost-benefit analysis, and accounting standards',
                'Layman': 'General public - Use simple language, avoid jargon, focus on practical implications',
                'Student': 'Academic learner - Provide educational context, explain concepts clearly, include learning objectives',
                'CEO': 'Chief Executive Officer - Focus on business impact, strategic decisions, and organizational implications',
                'Financial Product Consumer': 'Individual consumer - Focus on personal rights, practical steps, and consumer protection',
                'Legal Counsel': 'Lawyer - Emphasize legal precedents, case law, and detailed legal analysis',
                'IT Manager': 'Technology manager - Focus on technical implementation, system requirements, and operational aspects',
                'HR Manager': 'Human Resources - Focus on employee data, workplace policies, and HR compliance',
                'Marketing Manager': 'Marketing professional - Focus on customer data, marketing compliance, and business opportunities',
                'Small Business Owner': 'Entrepreneur - Focus on practical implementation, cost considerations, and business growth',
                'Developer': 'Software developer - Focus on technical implementation, code examples, and development practices',
                'Auditor': 'Compliance auditor - Focus on audit trails, evidence requirements, and compliance verification'
            }
            
            st.info(f"**{st.session_state.guardrails['persona']}**: {persona_descriptions.get(st.session_state.guardrails['persona'], 'Custom persona')}")
        
        # Custom prompt
        st.subheader("📝 Custom System Prompt")
        st.session_state.guardrails['custom_prompt'] = st.text_area(
            "Guardrail Prompt",
            value=st.session_state.guardrails['custom_prompt'],
            placeholder="You are a GDPR compliance expert. Only provide answers related to data protection, privacy rights, and regulatory compliance. If asked about topics outside GDPR, politely decline and redirect to relevant GDPR topics.",
            height=150,
            help="Custom prompt to guide the AI's behavior and responses"
        )
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            st.session_state.guardrails['max_response_length'] = st.slider(
                "📏 Maximum Response Length (tokens)",
                min_value=100,
                max_value=2000,
                value=st.session_state.guardrails['max_response_length'],
                step=100
            )
            
            st.session_state.guardrails['require_sources'] = st.checkbox(
                "📚 Require Document Sources",
                value=st.session_state.guardrails['require_sources'],
                help="Only provide answers when relevant document sources are found"
            )
            
            st.session_state.guardrails['min_relevance_score'] = st.slider(
                "🎯 Minimum Relevance Score",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.guardrails['min_relevance_score'],
                step=0.1,
                help="Minimum relevance score for document chunks to be used"
            )
            
            st.session_state.guardrails['safety_checks'] = st.checkbox(
                "🛡️ Enable Safety Checks",
                value=st.session_state.guardrails['safety_checks'],
                help="Enable additional safety and content filtering"
            )
    
    with col2:
        st.subheader("📋 Quick Templates")
        
        # Pre-defined prompt templates
        template_options = {
            "GDPR Expert": "You are a GDPR compliance expert. Only provide answers related to data protection, privacy rights, and regulatory compliance. If asked about topics outside GDPR, politely decline and redirect to relevant GDPR topics.",
            
            "Strict GDPR": "You are a GDPR compliance specialist. ONLY answer questions directly related to GDPR, data protection, privacy rights, and regulatory compliance. For any other topics, respond with: 'I can only assist with GDPR and data protection matters. Please ask a question related to data protection, privacy rights, or GDPR compliance.'",
            
            "Legal Advisor": "You are a legal advisor specializing in data protection law. Provide accurate, professional advice on GDPR compliance, data subject rights, and privacy regulations. Always recommend consulting with qualified legal counsel for specific legal matters.",
            
            "Compliance Officer": "You are a data protection compliance officer. Help with GDPR implementation, compliance procedures, and best practices. Focus on practical, actionable guidance for organizations handling personal data.",
            
            "Privacy Consultant": "You are a privacy consultant. Provide guidance on GDPR compliance, data protection impact assessments, and privacy by design principles. Emphasize practical implementation strategies.",
            
            "CISO Focus": "You are a GDPR expert responding to a Chief Information Security Officer. Focus on security risks, technical controls, threat mitigation, and strategic security planning related to data protection.",
            
            "DPO Focus": "You are a GDPR expert responding to a Data Protection Officer. Emphasize compliance requirements, legal obligations, regulatory frameworks, and audit requirements.",
            
            "Business Executive": "You are a GDPR expert responding to a business executive. Focus on business impact, strategic decisions, organizational implications, and competitive advantage of data protection compliance.",
            
            "Consumer Rights": "You are a GDPR expert responding to a consumer. Focus on personal rights, practical steps, consumer protection, and individual privacy rights in simple, clear language."
        }
        
        selected_template = st.selectbox(
            "Choose a template:",
            options=list(template_options.keys()),
            help="Select a pre-defined prompt template"
        )
        
        if st.button("📋 Apply Template"):
            st.session_state.guardrails['custom_prompt'] = template_options[selected_template]
            st.success(f"Applied {selected_template} template!")
            st.rerun()
        
        st.subheader("🔍 Current Settings")
        
        # Display current settings
        st.json({
            "LLM as Source": st.session_state.guardrails['use_llm_as_source'],
            "Domain": st.session_state.guardrails['domain_restriction'],
            "Tone": st.session_state.guardrails['response_tone'],
            "Persona": st.session_state.guardrails['persona'],
            "Strict Persona": st.session_state.guardrails['strict_persona'],
            "Max Length": st.session_state.guardrails['max_response_length'],
            "Require Sources": st.session_state.guardrails['require_sources'],
            "Min Relevance": st.session_state.guardrails['min_relevance_score'],
            "Safety Checks": st.session_state.guardrails['safety_checks']
        })
        
        # Test guardrails
        st.subheader("🧪 Test Guardrails")
        test_query = st.text_input(
            "Test Query:",
            placeholder="Ask a test question to see how guardrails work"
        )
        
        if st.button("🚀 Test") and test_query:
            with st.spinner("Testing guardrails..."):
                # Apply guardrails to the query
                result = rag_system.query_with_guardrails(
                    test_query, 
                    st.session_state.guardrails
                )
                display_query_results(result)
    
    # Save/Reset buttons
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            st.success("Guardrails configuration saved!")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.session_state.guardrails = {
                'use_llm_as_source': False,
                'custom_prompt': '',
                'domain_restriction': 'GDPR',
                'max_response_length': 1000,
                'require_sources': True,
                'min_relevance_score': 0.3,
                'safety_checks': True,
                'response_tone': 'Professional',
                'persona': 'None',
                'strict_persona': False,
                'persona_threshold': 0.2
            }
            st.success("Reset to default settings!")
            st.rerun()
    
    with col3:
        if st.button("📤 Export Config"):
            config_json = st.session_state.guardrails
            st.download_button(
                label="Download Configuration",
                data=str(config_json),
                file_name="guardrails_config.json",
                mime="application/json"
            )


def query_history_section():
    """Display query history"""
    if st.session_state.query_history:
        st.header("📜 Query History")
        
        # Display recent queries
        for i, history_item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {history_item['query'][:50]}..."):
                st.write(f"**Query:** {history_item['query']}")
                st.write(f"**Time:** {history_item['timestamp']}")
                
                if st.button("View Result", key=f"view_{i}"):
                    display_query_results(history_item["result"])


def main():
    """Main application function"""
    st.set_page_config(
        page_title="GDPR RAG System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🛡️ GDPR RAG System")
    st.markdown("""
    Upload GDPR documents and ask questions about compliance, data protection, and privacy regulations.
    This system uses AWS Bedrock for intelligent responses based on your uploaded documents.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment variables info
        st.subheader("🔑 Required Environment Variables")
        st.markdown("""
        **For AWS Bedrock:**
        - `AWS_ACCESS_KEY_ID`
        - `AWS_SECRET_ACCESS_KEY`
        - `AWS_REGION`
        
        **For Qdrant Cloud:**
        - `QDRANT_URL`
        - `QDRANT_API_KEY`
        """)
        
        # Initialize RAG system
        if st.button("🚀 Initialize System"):
            with st.spinner("Initializing RAG system..."):
                try:
                    st.session_state.rag_system = setup_rag_system()
                    st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing system: {str(e)}")
    
    # Main content
    if st.session_state.rag_system is None:
        st.warning("⚠️ Please initialize the system from the sidebar first.")
        return
    
    rag_system = st.session_state.rag_system
    
    # Display system status
    display_system_status(rag_system)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Documents", "🔍 Query Documents", "🛡️ Guardrails", "📜 History"])
    
    with tab1:
        upload_documents_section(rag_system)
    
    with tab2:
        query_section(rag_system)
    
    with tab3:
        guardrails_section(rag_system)
    
    with tab4:
        query_history_section()


if __name__ == "__main__":
    main()
