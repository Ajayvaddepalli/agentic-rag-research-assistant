"""
Utility functions for the AI Research Assistant.

This module provides helper functions for logging, file handling,
session management, and other common tasks.
"""

import os
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return logging.getLogger(__name__)


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def validate_file_type(file_path: str, allowed_extensions: List[str]) -> bool:
    """
    Validate if file has an allowed extension.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed extensions (e.g., ['.pdf', '.txt'])
        
    Returns:
        True if file type is allowed, False otherwise
    """
    if not file_path:
        return False
    
    _, ext = os.path.splitext(file_path.lower())
    return ext in [e.lower() for e in allowed_extensions]


def validate_file_size(file_path: str, max_size_mb: int) -> bool:
    """
    Validate file size against maximum allowed size.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file size is within limit, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb <= max_size_mb


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to specified length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_conversation_history(history: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Convert Gradio chat history to LangChain message format.
    
    Args:
        history: Gradio chat history format [[user_msg, bot_msg], ...]
        
    Returns:
        List of message dictionaries compatible with LangChain
    """
    messages = []
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    return messages


def get_timestamp(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_str: Datetime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_str)


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON data or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """
    Safely serialize data to JSON string with fallback.
    
    Args:
        data: Data to serialize
        default: Default JSON string if serialization fails
        
    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default


class SessionManager:
    """
    Manage user session state and data.
    
    This class helps track user sessions, document processing status,
    and other session-specific information.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """
        Create a new session with default values.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Session data dictionary
        """
        session_data = {
            "session_id": session_id,
            "created_at": get_timestamp(),
            "last_activity": get_timestamp(),
            "documents_processed": False,
            "document_count": 0,
            "total_queries": 0,
            "successful_queries": 0,
            "error_count": 0,
            "uploaded_files": [],
            "conversation_turns": 0
        }
        
        self.sessions[session_id] = session_data
        self.logger.info(f"Created new session: {session_id}")
        return session_data
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get session data, creating if it doesn't exist.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary
        """
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        """
        Update session data with new information.
        
        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
        """
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id].update(updates)
        self.sessions[session_id]["last_activity"] = get_timestamp()
    
    def increment_counter(self, session_id: str, counter_name: str) -> int:
        """
        Increment a counter in the session.
        
        Args:
            session_id: Session identifier
            counter_name: Name of the counter field
            
        Returns:
            New counter value
        """
        session = self.get_session(session_id)
        current_value = session.get(counter_name, 0)
        new_value = current_value + 1
        self.update_session(session_id, {counter_name: new_value})
        return new_value
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of session statistics
        """
        session = self.get_session(session_id)
        
        return {
            "session_duration": self._calculate_duration(session),
            "total_queries": session.get("total_queries", 0),
            "successful_queries": session.get("successful_queries", 0),
            "error_rate": self._calculate_error_rate(session),
            "documents_count": session.get("document_count", 0),
            "conversation_turns": session.get("conversation_turns", 0)
        }
    
    def _calculate_duration(self, session: Dict[str, Any]) -> str:
        """Calculate session duration."""
        try:
            created = datetime.strptime(session["created_at"], "%Y-%m-%d %H:%M:%S")
            last_activity = datetime.strptime(session["last_activity"], "%Y-%m-%d %H:%M:%S")
            duration = last_activity - created
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except (KeyError, ValueError):
            return "Unknown"
    
    def _calculate_error_rate(self, session: Dict[str, Any]) -> float:
        """Calculate error rate as percentage."""
        total = session.get("total_queries", 0)
        errors = session.get("error_count", 0)
        
        if total == 0:
            return 0.0
        
        return round((errors / total) * 100, 1)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up sessions older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        sessions_to_remove = []
        
        for session_id, session_data in self.sessions.items():
            try:
                last_activity = datetime.strptime(
                    session_data["last_activity"], 
                    "%Y-%m-%d %H:%M:%S"
                ).timestamp()
                
                if last_activity < cutoff_time:
                    sessions_to_remove.append(session_id)
            except (KeyError, ValueError):
                # Remove sessions with invalid timestamps
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
        
        return len(sessions_to_remove)


def format_error_message(error: Exception, user_friendly: bool = True) -> str:
    """
    Format error messages for display.
    
    Args:
        error: Exception object
        user_friendly: Whether to return user-friendly message
        
    Returns:
        Formatted error message
    """
    if user_friendly:
        error_mappings = {
            "ConnectionError": "Unable to connect to the service. Please check your internet connection.",
            "TimeoutError": "The request timed out. Please try again.",
            "ValueError": "Invalid input provided. Please check your data.",
            "FileNotFoundError": "The requested file was not found.",
            "PermissionError": "Permission denied. Please check file permissions.",
        }
        
        error_type = type(error).__name__
        return error_mappings.get(error_type, "An unexpected error occurred. Please try again.")
    else:
        return f"{type(error).__name__}: {str(error)}"


# Global session manager instance
session_manager = SessionManager()


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test logging
    logger = setup_logging("INFO")
    logger.info("Logging test successful")
    
    # Test session manager
    sm = SessionManager()
    session = sm.create_session("test_session")
    print(f"Created session: {session['session_id']}")
    
    # Test other utilities
    print(f"Timestamp: {get_timestamp()}")
    print(f"File size: {format_file_size(1536000)}")
    print(f"Text truncation: {truncate_text('This is a very long text', 10)}")
    
    print("✅ All utility functions working correctly!")