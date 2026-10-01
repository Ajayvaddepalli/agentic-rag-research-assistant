"""
Configuration management for AI Research Assistant.

This module handles all configuration settings including API keys,
model parameters, and application settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Application configuration management.
    """
    
    # Deployment mode detection
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")
    
    # LLM Configuration
    if DEPLOYMENT_MODE == "cloud":
        # Use Groq for free cloud deployment
        LLM_TYPE = "groq"
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for cloud deployment")
        # Groq model selection
        GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Fast free model
    else:
        # Use Ollama for local development
        LLM_TYPE = "ollama"
        OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

    # API Keys (set only what you use)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    
    # Optional: LangChain tracing (leave off unless using LangSmith)
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: Optional[str] = os.getenv("LANGCHAIN_API_KEY")

    # Model configuration
    # Keep OpenAI model for backward compatibility
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    # Embeddings:
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Temperature - adjusted for better Groq performance
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Max tokens - optimized for both Ollama and Groq
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Search Configuration
    WEB_SEARCH_RESULTS: int = int(os.getenv("WEB_SEARCH_RESULTS", "3"))
    DOCUMENT_SEARCH_RESULTS: int = int(os.getenv("DOCUMENT_SEARCH_RESULTS", "3"))
    
    # Gradio Configuration
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", "7860"))
    
    # Application Settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    SUPPORTED_FILE_TYPES: list = [".pdf"]
    
    @classmethod
    def get_llm_info(cls) -> dict:
        """Get current LLM configuration info."""
        if cls.LLM_TYPE == "groq":
            return {
                "type": "groq",
                "model": cls.GROQ_MODEL,
                "deployment": "cloud",
                "provider": "Groq"
            }
        elif cls.LLM_TYPE == "ollama":
            return {
                "type": "ollama", 
                "model": cls.OLLAMA_MODEL,
                "deployment": "local",
                "provider": "Ollama"
            }
        else:
            return {
                "type": "openai",
                "model": cls.LLM_MODEL,
                "deployment": "cloud",
                "provider": "OpenAI"
            }
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required configuration is present and valid.
        """
        errors = []
        
        # Validate based on LLM type
        if cls.LLM_TYPE == "groq":
            if not cls.GROQ_API_KEY:
                errors.append("GROQ_API_KEY is required for Groq deployment")
            elif not cls.GROQ_API_KEY.startswith("gsk_"):
                errors.append("GROQ_API_KEY should start with 'gsk_'")
                
        elif cls.LLM_TYPE == "ollama":
            if not cls.OLLAMA_BASE_URL:
                errors.append("OLLAMA_BASE_URL is required for Ollama deployment")
            if not cls.OLLAMA_MODEL:
                errors.append("OLLAMA_MODEL is required for Ollama deployment")
                
        elif cls.LLM_TYPE == "openai" or "gpt-" in cls.LLM_MODEL:
            if not cls.OPENAI_API_KEY:
                errors.append("OPENAI_API_KEY is required for OpenAI models")
        
        # Tavily key validation (optional but recommended)
        if not cls.TAVILY_API_KEY:
            print("Warning: TAVILY_API_KEY not set - web search will be disabled")
        
        # Validate numeric settings
        if not (0 <= cls.TEMPERATURE <= 2):
            errors.append("TEMPERATURE must be between 0 and 2")
            
        if not (100 <= cls.CHUNK_SIZE <= 5000):
            errors.append("CHUNK_SIZE must be between 100 and 5000")
            
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
            
        if not (1 <= cls.WEB_SEARCH_RESULTS <= 10):
            errors.append("WEB_SEARCH_RESULTS must be between 1 and 10")
            
        if not (1 <= cls.DOCUMENT_SEARCH_RESULTS <= 10):
            errors.append("DOCUMENT_SEARCH_RESULTS must be between 1 and 10")
            
        if not (512 <= cls.MAX_TOKENS <= 4096):
            errors.append("MAX_TOKENS must be between 512 and 4096")
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_message)
            
        return True
    
    @classmethod
    def get_model_config(cls) -> dict:
        """Get model configuration."""
        base_config = {
            "embedding_model": cls.EMBEDDING_MODEL,
            "hf_embedding_model": cls.HF_EMBEDDING_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
        }
        
        # Add LLM-specific config
        if cls.LLM_TYPE == "groq":
            base_config.update({
                "llm_model": cls.GROQ_MODEL,
                "llm_type": "groq",
                "api_key": cls.GROQ_API_KEY
            })
        elif cls.LLM_TYPE == "ollama":
            base_config.update({
                "llm_model": cls.OLLAMA_MODEL,
                "llm_type": "ollama",
                "base_url": cls.OLLAMA_BASE_URL
            })
        else:
            base_config.update({
                "llm_model": cls.LLM_MODEL,
                "llm_type": "openai",
                "api_key": cls.OPENAI_API_KEY
            })
            
        return base_config
    
    @classmethod
    def get_search_config(cls) -> dict:
        return {
            "web_search_results": cls.WEB_SEARCH_RESULTS,
            "document_search_results": cls.DOCUMENT_SEARCH_RESULTS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
        }
    
    @classmethod
    def get_app_config(cls) -> dict:
        return {
            "gradio_port": cls.GRADIO_PORT,
            "gradio_share": cls.GRADIO_SHARE,
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "supported_file_types": cls.SUPPORTED_FILE_TYPES,
            "vector_db_path": cls.VECTOR_DB_PATH,
            "deployment_mode": cls.DEPLOYMENT_MODE,
        }
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration."""
        llm_info = cls.get_llm_info()
        
        print("🚀 NEXUS AI Research Assistant Configuration")
        print("=" * 50)
        print(f"🤖 LLM Provider: {llm_info['provider']}")
        print(f"📡 Model: {llm_info['model']}")
        print(f"🌍 Deployment: {llm_info['deployment']}")
        print(f"🌡️  Temperature: {cls.TEMPERATURE}")
        print(f"🔤 Max Tokens: {cls.MAX_TOKENS}")
        print(f"📄 Chunk Size: {cls.CHUNK_SIZE}")
        print(f"🔄 Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"🔍 Web Search Results: {cls.WEB_SEARCH_RESULTS}")
        print(f"📚 Document Search Results: {cls.DOCUMENT_SEARCH_RESULTS}")
        print(f"🌐 Gradio Port: {cls.GRADIO_PORT}")
        print(f"💾 Vector DB Path: {cls.VECTOR_DB_PATH}")
        print("-" * 50)
        print("🔑 API Keys Status:")
        
        if cls.LLM_TYPE == "groq":
            print(f"   Groq API Key: {'✅ Set' if cls.GROQ_API_KEY else '❌ Missing'}")
        elif cls.LLM_TYPE == "ollama":
            print(f"   Ollama URL: {'✅ Set' if cls.OLLAMA_BASE_URL else '❌ Missing'}")
        else:
            print(f"   OpenAI API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Missing'}")
            
        print(f"   Tavily API Key: {'✅ Set' if cls.TAVILY_API_KEY else '⚠️  Missing (web search disabled)'}")
        print("=" * 50)

# Validate configuration on import
try:
    Config.validate()
    print("✅ Configuration validated successfully!")
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")
    print("Please check your .env file and ensure required keys are set.")
    print("Copy .env.example to .env and fill in your API keys.")

if __name__ == "__main__":
    try:
        Config.validate()
        Config.print_config()
        print("\n🎉 Configuration is valid and ready!")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
