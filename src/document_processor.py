"""
Document processing module for AI Research Assistant.

This module handles PDF loading, text chunking, vector database operations,
and document retrieval functionality.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# Loaders and community integrations (v0.2 migration)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# Core LangChain utilities that remain in langchain.*
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


from .config import Config
from .utils import get_file_hash, format_file_size, validate_file_type, validate_file_size


class DocumentProcessor:
    """
    Handles document processing and vector database operations.
    
    This class provides functionality for:
    - Loading PDF documents
    - Chunking text into optimal sizes
    - Creating and managing vector embeddings
    - Storing documents in ChromaDB
    - Retrieving relevant documents for queries
    """
    
    def __init__(self, vector_db_path: Optional[str] = None):
        """
        Initialize the document processor.
        
        Args:
            vector_db_path: Custom path for vector database storage
        """
        self.vector_db_path = vector_db_path or Config.VECTOR_DB_PATH
        self.logger = logging.getLogger(__name__)
        
        # Initialize embeddings with configured model
        try:
            self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

        except Exception as e:
            self.logger.error(f"Failed to initialize embeddings: {e}")
            raise
        
        # Configure text splitter for optimal chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\\n\\n", "\\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize storage variables
        self.vectorstore = None
        self.retriever = None
        self.processed_files = {}  # Track processed files
        
        # Ensure vector database directory exists
        Path(self.vector_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DocumentProcessor initialized with vector DB path: {self.vector_db_path}")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a file before processing.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": False,
            "file_path": file_path,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        # Check if file exists
        if not os.path.exists(file_path):
            validation_result["errors"].append(f"File not found: {file_path}")
            return validation_result
        
        # Get file info
        file_size = os.path.getsize(file_path)
        validation_result["file_info"] = {
            "name": os.path.basename(file_path),
            "size_bytes": file_size,
            "size_formatted": format_file_size(file_size),
            "extension": os.path.splitext(file_path)[1].lower()
        }
        
        # Validate file type
        if not validate_file_type(file_path, Config.SUPPORTED_FILE_TYPES):
            validation_result["errors"].append(
                f"Unsupported file type. Supported types: {Config.SUPPORTED_FILE_TYPES}"
            )
        
        # Validate file size
        if not validate_file_size(file_path, Config.MAX_FILE_SIZE_MB):
            validation_result["errors"].append(
                f"File too large. Maximum size: {Config.MAX_FILE_SIZE_MB}MB"
            )
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except Exception as e:
            validation_result["errors"].append(f"Cannot read file: {str(e)}")
        
        # Set valid flag
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF document and extract text with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects with content and metadata
            
        Raises:
            Exception: If PDF loading fails
        """
        try:
            # Validate file first
            validation = self.validate_file(file_path)
            if not validation["valid"]:
                raise ValueError(f"File validation failed: {validation['errors']}")
            
            self.logger.info(f"Loading PDF: {file_path}")
            
            # Load PDF using PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Calculate file hash for tracking
            file_hash = get_file_hash(file_path)
            
            # Enhance metadata for each document
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "file_size": validation["file_info"]["size_bytes"],
                    "page_number": i + 1,
                    "total_pages": len(documents),
                    "processed_at": datetime.now().isoformat(),
                    "chunk_method": "recursive_character",
                    "embedding_model": Config.EMBEDDING_MODEL
                })
            
            # Track processed file
            self.processed_files[file_path] = {
                "hash": file_hash,
                "pages": len(documents),
                "processed_at": datetime.now().isoformat(),
                "status": "loaded"
            }
            
            self.logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise Exception(f"Failed to load PDF: {str(e)}")
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into optimally-sized chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of document chunks
        """
        try:
            self.logger.info(f"Chunking {len(documents)} documents...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content),
                    "chunked_at": datetime.now().isoformat()
                })
            
            self.logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking documents: {str(e)}")
            raise Exception(f"Failed to chunk documents: {str(e)}")
    
    def create_or_load_vectorstore(self) -> None:
        """
        Create a new vector store or load existing one.
        
        Raises:
            Exception: If vector store creation/loading fails
        """
        try:
            if os.path.exists(self.vector_db_path):
                # Load existing vector store
                self.logger.info(f"Loading existing vector store from {self.vector_db_path}")
                self.vectorstore = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embeddings
                )
                self.logger.info("Existing vector store loaded successfully")
            else:
                # Will create new vector store when documents are added
                self.logger.info("No existing vector store found. Will create new one when documents are processed.")
                self.vectorstore = None
                
        except Exception as e:
            self.logger.error(f"Error with vector store: {str(e)}")
            raise Exception(f"Vector store error: {str(e)}")
    
    def add_documents_to_vectorstore(self, chunks: List[Document]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks to add
            
        Raises:
            Exception: If adding documents fails
        """
        try:
            if self.vectorstore is None:
                # Create new vector store
                self.logger.info("Creating new vector store...")
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_path
                )
                self.logger.info(f"Created new vector store with {len(chunks)} chunks")
            else:
                # Add to existing vector store
                self.logger.info(f"Adding {len(chunks)} chunks to existing vector store...")
                self.vectorstore.add_documents(chunks)
                self.logger.info("Documents added to existing vector store")
            
            # Persist the changes
            self.vectorstore.persist()
            self.logger.info("Vector store changes persisted")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {str(e)}")
            raise Exception(f"Failed to add documents to vector store: {str(e)}")
    
    def create_retriever(self) -> None:
        """
        Create a retriever from the vector store.
        
        Raises:
            Exception: If retriever creation fails
        """
        try:
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized. Process documents first.")
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": Config.DOCUMENT_SEARCH_RESULTS,
                    "filter": None  # Can add metadata filters here
                }
            )
            self.logger.info("Document retriever created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating retriever: {str(e)}")
            raise Exception(f"Failed to create retriever: {str(e)}")
    
    def process_documents(self, file_paths: List[str]) -> bool:
        """
        Process multiple PDF documents end-to-end.
        
        This is the main method that orchestrates the entire document processing pipeline:
        1. Validate files
        2. Load PDFs
        3. Chunk documents
        4. Create/update vector store
        5. Create retriever
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            if not file_paths:
                self.logger.warning("No files provided for processing")
                return False
            
            self.logger.info(f"Starting document processing for {len(file_paths)} files...")
            
            # Step 1: Validate all files first
            all_documents = []
            validation_results = []
            
            for file_path in file_paths:
                validation = self.validate_file(file_path)
                validation_results.append(validation)
                
                if not validation["valid"]:
                    self.logger.error(f"File validation failed for {file_path}: {validation['errors']}")
                    continue
                
                # Step 2: Load PDF
                documents = self.load_pdf(file_path)
                all_documents.extend(documents)
            
            if not all_documents:
                self.logger.error("No valid documents loaded")
                return False
            
            # Step 3: Chunk all documents
            chunks = self.chunk_documents(all_documents)
            
            # Step 4: Create or load vector store
            self.create_or_load_vectorstore()
            
            # Step 5: Add documents to vector store
            self.add_documents_to_vectorstore(chunks)
            
            # Step 6: Create retriever
            self.create_retriever()
            
            # Update processing status
            for file_path in file_paths:
                if file_path in self.processed_files:
                    self.processed_files[file_path]["status"] = "processed"
            
            self.logger.info(f"Successfully processed {len(file_paths)} files with {len(chunks)} total chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in document processing pipeline: {str(e)}")
            return False
    
    def get_retriever(self):
        """
        Get the document retriever, loading existing vector store if needed.
        
        Returns:
            Document retriever or None if no documents processed
        """
        if self.retriever is None:
            try:
                # Try to load existing vector store
                self.create_or_load_vectorstore()
                if self.vectorstore is not None:
                    self.create_retriever()
            except Exception as e:
                self.logger.warning(f"Could not load existing vector store: {e}")
                return None
        
        return self.retriever
    
    def search_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Search documents for relevant content.
        
        Args:
            query: Search query string
            k: Number of results to return (uses config default if not specified)
            
        Returns:
            List of relevant document chunks
        """
        retriever = self.get_retriever()
        if retriever is None:
            self.logger.warning("No retriever available. Process documents first.")
            return []
        
        try:
            # Use custom k or default from config
            search_k = k or Config.DOCUMENT_SEARCH_RESULTS
            
            # Update retriever search parameters
            retriever.search_kwargs["k"] = search_k
            
            # Perform search
            results = retriever.get_relevant_documents(query)
            
            self.logger.info(f"Found {len(results)} relevant chunks for query: '{query[:50]}...'")
            return results[:search_k]  # Ensure we don't return more than requested
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current vector database.
        
        Returns:
            Dictionary with database information and statistics
        """
        info = {
            "database_path": self.vector_db_path,
            "database_exists": os.path.exists(self.vector_db_path),
            "vectorstore_loaded": self.vectorstore is not None,
            "retriever_available": self.retriever is not None,
            "embedding_model": Config.EMBEDDING_MODEL,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "processed_files": len(self.processed_files),
            "files_info": self.processed_files.copy()
        }
        
        # Add vector store specific info if available
        if self.vectorstore is not None:
            try:
                # Try to get collection info (ChromaDB specific)
                info["status"] = "Ready"
                info["last_updated"] = datetime.now().isoformat()
            except Exception as e:
                info["status"] = f"Error accessing vector store: {str(e)}"
        else:
            info["status"] = "No vector store loaded"
        
        return info
    
    def clear_database(self) -> bool:
        """
        Clear the vector database and reset the processor.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Clearing vector database...")
            
            # Reset internal state
            self.vectorstore = None
            self.retriever = None
            self.processed_files = {}
            
            # Remove database files
            if os.path.exists(self.vector_db_path):
                import shutil
                shutil.rmtree(self.vector_db_path)
                self.logger.info(f"Removed vector database directory: {self.vector_db_path}")
            
            self.logger.info("Vector database cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def get_processed_files_summary(self) -> List[Dict[str, Any]]:
        """
        Get a summary of all processed files.
        
        Returns:
            List of file information dictionaries
        """
        summary = []
        for file_path, info in self.processed_files.items():
            file_info = {
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "pages": info.get("pages", 0),
                "processed_at": info.get("processed_at", "Unknown"),
                "status": info.get("status", "Unknown"),
                "hash": info.get("hash", "Unknown")[:8] + "..."  # Show first 8 chars of hash
            }
            summary.append(file_info)
        
        return summary


def test_document_processor():
    """Test the document processor with basic functionality."""
    print("🧪 Testing DocumentProcessor...")
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        print("✅ DocumentProcessor initialized")
        
        # Test database info
        info = processor.get_database_info()
        print(f"✅ Database info retrieved: {info['status']}")
        
        # Test search without documents
        results = processor.search_documents("test query")
        print(f"✅ Search test completed (found {len(results)} results)")
        
        print("🎉 DocumentProcessor test completed successfully!")
        return processor
        
    except Exception as e:
        print(f"❌ DocumentProcessor test failed: {str(e)}")
        return None


if __name__ == "__main__":
    test_document_processor()