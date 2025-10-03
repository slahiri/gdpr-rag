# GDPR RAG System

A Retrieval-Augmented Generation (RAG) system for GDPR document analysis with a Streamlit UI and AWS Bedrock integration.

## Features

- 📄 **PDF Upload**: Upload GDPR-related PDF documents
- 🔍 **Intelligent Querying**: Ask questions about GDPR compliance and get AI-powered answers
- 📊 **Relevance Scoring**: See how relevant each source document is to your query
- 🛡️ **GDPR-Focused**: Specialized prompts for GDPR compliance questions
- ☁️ **AWS Bedrock Integration**: Uses Claude models for high-quality responses
- 🗄️ **Vector Database Support**: ChromaDB (local) and Pinecone (cloud) support

## Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project directory
cd /Users/sid/rag

# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done)
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file based on `env_example.txt`:

```bash
cp env_example.txt .env
```

Edit `.env` with your credentials:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1

# Vector Database Configuration (choose one)
# For Pinecone
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# For Chroma (local) - no additional config needed
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Bedrock Model Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Vector Database: Qdrant Cloud

This system uses **Qdrant Cloud** for vector storage and similarity search:

- ✅ Fast and efficient vector search
- ✅ Scalable cloud infrastructure
- ✅ Excellent performance for RAG applications
- ✅ Secure and reliable

### Qdrant Configuration:
The system is pre-configured to use your Qdrant cloud instance:
- **URL**: Your cloud instance URL
- **API Key**: Your authentication key
- **Collection**: `gdpr_docs` (automatically created)

## Usage

### 1. Upload Documents
- Click "Upload Documents" tab
- Upload PDF files containing GDPR information
- Click "Process" to add documents to the vector database

### 2. Query Documents
- Click "Query Documents" tab
- Enter your GDPR-related question
- Click "Search" to get AI-powered answers
- View relevance scores and source documents

### 3. View History
- Access previous queries and results
- Re-run previous queries

## Example Queries

- "What are the key requirements for data processing under GDPR?"
- "What are the penalties for GDPR violations?"
- "How should I handle data subject requests?"
- "What is the difference between a data controller and data processor?"
- "What are the requirements for data breach notification?"

## System Architecture

```
PDF Upload → Text Extraction → Chunking → Vector Embeddings → Qdrant Cloud
                                                                    ↓
User Query → Query Embedding → Vector Search → Context Retrieval → AWS Bedrock → Response
```

## Components

- **PDF Processor**: Extracts and chunks text from PDF documents
- **Qdrant Vector Store**: Manages document embeddings and similarity search in the cloud
- **Bedrock Client**: Handles AWS Bedrock API interactions with Claude Sonnet 4
- **RAG System**: Orchestrates the complete pipeline
- **Streamlit UI**: User interface for document upload and querying

## Troubleshooting

### Common Issues

1. **AWS Bedrock Access Denied**
   - Ensure your AWS credentials have Bedrock permissions
   - Check that the model ID is available in your region

2. **Qdrant Connection Issues**
   - Verify your Qdrant URL and API key in the `.env` file
   - Check that your Qdrant cloud instance is running

3. **PDF Processing Errors**
   - Ensure PDF files are not password-protected
   - Check that PDFs contain extractable text (not just images)

### System Status

The application includes a system status panel that shows:
- Vector database connection status
- Document count
- AWS Bedrock connection status

## Development

### Project Structure

```
rag/
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py      # PDF text extraction and chunking
│   ├── vector_store.py       # Vector database operations
│   ├── bedrock_client.py     # AWS Bedrock integration
│   └── rag_system.py         # Main RAG orchestrator
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── env_example.txt          # Environment variables template
└── README.md                # This file
```

### Adding New Features

1. **New Vector Database**: Extend `VectorStore` class
2. **New LLM Provider**: Extend `BedrockClient` class
3. **New Document Types**: Extend `PDFProcessor` class

## License

This project is for educational and development purposes. Ensure compliance with AWS Bedrock terms of service and any applicable data protection regulations when using with real GDPR documents.
