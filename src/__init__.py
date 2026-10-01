"""
AI Research Assistant - Agentic RAG System

A sophisticated AI assistant that combines document retrieval with web search
using LangGraph agents for intelligent decision making.

This package provides:
- AgenticRAGAssistant: Main agent implementation
- DocumentProcessor: PDF processing and vector storage
- ToolManager: Agent tools for search capabilities
- Config: Configuration management

Example usage:
    from src import AgenticRAGAssistant, DocumentProcessor
    
    # Create document processor
    doc_processor = DocumentProcessor()
    doc_processor.process_documents(["document.pdf"])
    
    # Create agent
    assistant = AgenticRAGAssistant(doc_processor)
    
    # Chat with the assistant
    response = assistant.chat("What's in the document?")
    print(response)

Author: AI Research Assistant Team
Version: 1.0.0
License: MIT
"""

from .config import Config
from .document_processor import DocumentProcessor
from .tools import ToolManager
from .agent import AgenticRAGAssistant

__version__ = "1.0.0"
__author__ = "AI Research Assistant Team"
__license__ = "MIT"

__all__ = [
    "Config",
    "DocumentProcessor",
    "ToolManager", 
    "AgenticRAGAssistant"
]