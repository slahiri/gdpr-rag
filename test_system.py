"""
Test script for GDPR RAG System
Run this to test the system components
"""

import os
import sys
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_system import RAGSystem


def test_system():
    """Test the RAG system components"""
    print("🧪 Testing GDPR RAG System...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize system
    print("📋 Initializing RAG system...")
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "collection_name": "gdpr_docs"
    }
    
    try:
        rag_system = RAGSystem(config)
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False
    
    # Test system status
    print("\n📊 Checking system status...")
    status = rag_system.get_system_status()
    
    print(f"Vector Store: {status['vector_store']}")
    print(f"Bedrock: {status['bedrock']}")
    print(f"System Ready: {status['system_ready']}")
    
    if not status['system_ready']:
        print("⚠️ System not ready - check your AWS credentials and Bedrock access")
        return False
    
    # Test with sample text (simulating a PDF upload)
    print("\n📄 Testing document processing...")
    
    # Create a sample document
    sample_text = """
    GDPR Article 5 - Principles relating to processing of personal data
    
    1. Personal data shall be:
    (a) processed lawfully, fairly and in a transparent manner in relation to the data subject ('lawfulness, fairness and transparency');
    (b) collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes ('purpose limitation');
    (c) adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed ('data minimisation');
    (d) accurate and, where necessary, kept up to date ('accuracy');
    (e) kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed ('storage limitation');
    (f) processed in a manner that ensures appropriate security of the personal data ('integrity and confidentiality').
    
    2. The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1 ('accountability').
    """
    
    # Create a mock PDF file object
    class MockPDFFile:
        def __init__(self, content):
            self.content = content
            self.name = "sample_gdpr_article.pdf"
        
        def read(self):
            return self.content.encode()
        
        def seek(self, position):
            pass
    
    mock_pdf = MockPDFFile(sample_text)
    
    try:
        # Process the document
        result = rag_system.upload_document(mock_pdf, "sample_gdpr_article.pdf")
        print(f"Document processing result: {result}")
        
        if result["success"]:
            print("✅ Document processed successfully")
        else:
            print(f"❌ Document processing failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return False
    
    # Test querying
    print("\n🔍 Testing query functionality...")
    
    test_queries = [
        "What are the main principles of GDPR?",
        "What is data minimisation?",
        "What are the accountability requirements?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_system.query(query, top_k=3)
            
            if result["success"]:
                print(f"✅ Answer: {result['answer'][:200]}...")
                print(f"Sources found: {len(result['sources'])}")
                if result['scores']:
                    print(f"Average relevance score: {sum(result['scores'])/len(result['scores']):.3f}")
            else:
                print(f"❌ Query failed: {result['answer']}")
                
        except Exception as e:
            print(f"❌ Error during query: {e}")
    
    print("\n🎉 System test completed!")
    return True


if __name__ == "__main__":
    success = test_system()
    if success:
        print("\n✅ All tests passed! You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check the configuration and try again.")
