"""
Test Qdrant connection with your cloud instance
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

def test_qdrant_connection():
    """Test connection to Qdrant cloud instance"""
    print("🧪 Testing Qdrant Cloud Connection...")
    
    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment
    url = os.getenv("QDRANT_URL", "https://e5d30163-f1b1-4102-8dfc-063324f99e21.us-east-1-1.aws.cloud.qdrant.io:6333")
    api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.hBl6haNcri4n3c9C8DgnZWnRzCW5GEq0cTRjKU5gGTo")
    
    print(f"URL: {url}")
    print(f"API Key: {api_key[:20]}...")
    
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(url=url, api_key=api_key)
        
        # Test connection by getting collections
        collections = qdrant_client.get_collections()
        print(f"✅ Connection successful!")
        print(f"📋 Available collections: {collections}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    if success:
        print("\n🎉 Qdrant cloud connection is working!")
    else:
        print("\n❌ Qdrant connection failed. Please check your credentials.")

