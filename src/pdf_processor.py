"""
PDF Processing Module for GDPR RAG System
Handles PDF text extraction and document chunking
"""

import io
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class PDFProcessor:
    """Handles PDF processing and text extraction for GDPR documents"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with text splitting configuration
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from uploaded PDF file
        
        Args:
            pdf_file: Uploaded PDF file object
            
        Returns:
            Extracted text as string
        """
        try:
            # Read PDF from file object
            pdf_reader = PdfReader(pdf_file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text
                    text += "\n"
            
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def create_documents(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into documents with metadata
        
        Args:
            text: Extracted text from PDF
            metadata: Additional metadata for documents
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                "chunk_id": i,
                "source": metadata.get("source", "uploaded_pdf"),
                "total_chunks": len(chunks),
                **metadata
            }
            
            document = Document(
                page_content=chunk,
                metadata=doc_metadata
            )
            documents.append(document)
        
        return documents
    
    def process_pdf(self, pdf_file, filename: str = None) -> List[Document]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_file: Uploaded PDF file object
            filename: Name of the PDF file
            
        Returns:
            List of processed Document objects
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)
        
        # Create metadata
        import datetime
        metadata = {
            "filename": filename or "unknown.pdf",
            "document_type": "gdpr_document",
            "processing_timestamp": str(datetime.datetime.now())
        }
        
        # Create documents
        documents = self.create_documents(text, metadata)
        
        return documents


# Import datetime for timestamp
import datetime
