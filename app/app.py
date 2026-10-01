"""
Gradio web interface for the AI Research Assistant.

This module creates a modern, user-friendly web interface using Gradio
for interacting with the Agentic RAG system.
"""

import os
import sys
import gradio as gr
import logging
import traceback
from typing import List, Tuple, Optional, Dict, Any
import uuid
import json

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.agent import AgenticRAGAssistant
    from src.document_processor import DocumentProcessor
    from src.config import Config
    from src.utils import setup_logging, SessionManager, format_file_size, get_timestamp
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    print("And that all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Set up logging
logger = setup_logging("INFO", "app.log")

# Global variables for application state
assistant = None
session_manager = SessionManager()
current_session_id = "main_session"

def normalize_history_to_pairs(history):
    if not history:
        return []
    if isinstance(history[0], dict):
        pairs = []
        pending_user = None
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                pairs.append([pending_user, content])
                pending_user = None
        return pairs
    return history

def pairs_to_messages(history_pairs):
    messages = []
    for user_msg, bot_msg in history_pairs:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    return messages

def initialize_assistant() -> Optional[AgenticRAGAssistant]:
    global assistant
    try:
        logger.info("Initializing AI Research Assistant...")
        assistant = AgenticRAGAssistant()
        logger.info("Assistant initialized successfully")
        return assistant
    except Exception as e:
        logger.error(f"Failed to initialize assistant: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def validate_uploaded_files(files: List[str]) -> Tuple[List[str], List[str], str]:
    if not files:
        return [], [], "No files uploaded."
    
    valid_files, errors = [], []
    for file_path in files:
        try:
            if not os.path.exists(file_path):
                errors.append(f"File not found: {os.path.basename(file_path)}")
                continue
            if not file_path.lower().endswith('.pdf'):
                errors.append(f"Only PDF files are supported: {os.path.basename(file_path)}")
                continue
            file_size = os.path.getsize(file_path)
            if file_size > Config.MAX_FILE_SIZE_MB * 1024 * 1024:
                errors.append(f"File too large (max {Config.MAX_FILE_SIZE_MB}MB): {os.path.basename(file_path)}")
                continue
            valid_files.append(file_path)
            logger.info(f"File validated: {os.path.basename(file_path)} ({format_file_size(file_size)})")
        except Exception as e:
            errors.append(f"Error validating {os.path.basename(file_path)}: {str(e)}")
    
    if valid_files and not errors:
        status = f"{len(valid_files)} file(s) ready for processing"
    elif valid_files and errors:
        status = f"{len(valid_files)} valid, {len(errors)} errors"
    elif errors:
        status = f"{len(errors)} validation error(s)"
    else:
        status = "No files to process"
    return valid_files, errors, status

def process_uploaded_files(files: List[str]) -> Tuple[str, bool, Dict[str, Any]]:
    global assistant
    try:
        valid_files, errors, _ = validate_uploaded_files(files)
        if not valid_files:
            error_msg = "No valid files to process."
            if errors:
                error_msg += f" Errors: {'; '.join(errors)}"
            return error_msg, False, {"errors": errors}

        logger.info(f"Processing {len(valid_files)} valid files...")
        doc_processor = DocumentProcessor()
        start_time = get_timestamp()
        success = doc_processor.process_documents(valid_files)
        end_time = get_timestamp()

        if success:
            if assistant:
                assistant.update_document_processor(doc_processor)
            else:
                logger.warning("Assistant not initialized, cannot update document processor")

            db_info = doc_processor.get_database_info()
            file_names = [os.path.basename(f) for f in valid_files]

            success_msg = f"Successfully processed {len(valid_files)} files.\n"
            success_msg += f"Files: {', '.join(file_names)}\n"
            success_msg += f"Processing time: {start_time} - {end_time}\n"
            success_msg += f"Total processed files: {db_info.get('processed_files', 0)}"
            if errors:
                success_msg += f"\nSome files had errors: {'; '.join(errors)}"

            processing_info = {
                "success": True,
                "files_processed": len(valid_files),
                "file_names": file_names,
                "errors": errors,
                "database_info": db_info,
                "processing_time": f"{start_time} - {end_time}",
            }
            logger.info(f"Successfully processed {len(valid_files)} files")
            return success_msg, True, processing_info
        else:
            error_msg = "Failed to process documents. Please check the files and try again."
            if errors:
                error_msg += f"\nValidation errors: {'; '.join(errors)}"
            return error_msg, False, {"errors": errors, "processing_failed": True}
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        logger.error(traceback.format_exc())
        error_msg = f"Processing error: {str(e)}"
        return error_msg, False, {"exception": str(e)}

def chat_function(
    message: str,
    history: List[List[str]],
    files: Optional[List[str]] = None
) -> Tuple[str, List[Dict[str, str]], str, str]:
    global assistant, current_session_id

    # Normalize incoming history to pairs internally
    history = normalize_history_to_pairs(history)

    # Handle file uploads first
    file_status = "Ready"
    if files:
        try:
            msg, success, _ = process_uploaded_files(files)
            file_status = msg
            if not success:
                return "", pairs_to_messages(history), file_status, get_system_info()
        except Exception as e:
            logger.error(f"File processing error: {str(e)}")
            file_status = f"Error: {str(e)}"
            return "", pairs_to_messages(history), file_status, get_system_info()

    # Handle empty message
    if not message or not message.strip():
        return "", pairs_to_messages(history), file_status, get_system_info()

    # Initialize assistant if needed
    if not assistant:
        try:
            assistant = initialize_assistant()
            if not assistant:
                error_response = ("System Error: The AI assistant could not be initialized. "
                                  "Please check configuration.")
                history.append([message, error_response])
                return "", pairs_to_messages(history), "Initialization failed", get_system_info()
        except Exception as e:
            logger.error(f"Assistant initialization error: {str(e)}")
            error_response = f"Initialization Error: {str(e)}"
            history.append([message, error_response])
            return "", pairs_to_messages(history), "System error", get_system_info()

    try:
        session_manager.increment_counter(current_session_id, "total_queries")
        logger.info(f"Processing message: '{message[:50]}...'")
        response = assistant.chat(message, current_session_id)

        history.append([message, response])
        session_manager.increment_counter(current_session_id, "successful_queries")

        status_message = f"Response generated ({len(response)} chars)"
        logger.info(f"Generated response: {len(response)} characters")

        return "", pairs_to_messages(history), status_message, get_system_info()

    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        logger.error(traceback.format_exc())
        session_manager.increment_counter(current_session_id, "error_count")
        error_response = (
            "An error occurred while processing your message. "
            f"Details: {str(e)}"
        )
        history.append([message, error_response])
        return "", pairs_to_messages(history), f"Error: {str(e)[:60]}...", get_system_info()

def clear_chat() -> Tuple[List, str, str]:
    global current_session_id, assistant
    try:
        current_session_id = f"session_{uuid.uuid4().hex[:8]}"
        session_manager.create_session(current_session_id)
        if assistant:
            assistant.clear_conversation(current_session_id)
        logger.info(f"Chat cleared, new session: {current_session_id}")
        return [], "Chat cleared", get_system_info()
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return [], f"Clear error: {str(e)}", get_system_info()

def get_system_info() -> str:
    """
    Get current system information for display.
    """
    try:
        if not assistant:
            return "**System Status**: Assistant not initialized"

        # Get system status from assistant
        status = assistant.get_system_status()
        # Get session stats
        session_stats = session_manager.get_session_stats(current_session_id)

        # Format information
        info_lines = [
            "**🚀 System Status**",
            f"- **Agent**: {'Online' if status.get('agent_initialized') else 'Offline'}",
            f"- **Model**: {status.get('llm_model', 'Unknown')}",
            f"- **Documents**: {'Available' if status.get('documents_available') else 'None uploaded'}",
            f"- **Tools**: {len(status.get('available_tools', []))} available",
            "",
            "**📊 Mission Stats**",
            f"- **Queries**: {session_stats.get('total_queries', 0)} total, {session_stats.get('successful_queries', 0)} successful",
            f"- **Duration**: {session_stats.get('session_duration', 'Unknown')}",
            f"- **Error Rate**: {session_stats.get('error_rate', 0)}%",
            "",
            f"**⏰ Last Updated**: {get_timestamp()}"
        ]

        return "\n".join(info_lines)

    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return f"**System Status**: Error: {str(e)}"

def create_interface():
    """
    Create and configure the Gradio interface with improved desktop layout alignment.
    """
    # Improved Space theme CSS with better desktop alignment
    css = """
    /* Space Theme Styling - Enhanced Desktop Layout */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
    
    .gradio-container {
        max-width: 100vw !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        font-family: 'Exo 2', 'Segoe UI', sans-serif !important;
        background: radial-gradient(ellipse at center, #0f0f23 0%, #050511 100%) !important;
        min-height: 100vh !important;
        color: #e0e6ed !important;
        position: relative;
        overflow-x: hidden !important;
    }

    /* Force proper desktop grid layout */
    .gradio-container > .gr-row {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Main content row - force horizontal layout */
    .main-content-row {
        display: grid !important;
        grid-template-columns: 1fr 400px !important;
        gap: 20px !important;
        padding: 20px !important;
        width: calc(100vw - 40px) !important;
        max-width: none !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }

    /* Ensure no responsive breaking */
    @media (max-width: 1400px) {
        .main-content-row {
            grid-template-columns: 1fr 350px !important;
        }
    }

    @media (max-width: 1200px) {
        .main-content-row {
            grid-template-columns: 1fr 320px !important;
        }
    }

    /* Left panel - chat area */
    .left-content {
        display: flex !important;
        flex-direction: column !important;
        gap: 15px !important;
        min-width: 0 !important; /* Prevent overflow */
    }

    /* Right panel - controls */
    .right-content {
        display: flex !important;
        flex-direction: column !important;
        gap: 15px !important;
        width: 100% !important;
        min-width: 320px !important;
        max-width: 400px !important;
    }

    /* Animated stars background */
    .gradio-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #fff, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 30px, #fff, transparent);
        background-size: 200px 100px;
        animation: twinkle 20s infinite linear;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes twinkle {
        0% { transform: translateX(0); }
        100% { transform: translateX(-200px); }
    }

    /* Header styling */
    .space-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 30px 20px !important;
        margin: 0 !important;
        border-radius: 0 0 30px 30px;
        box-shadow: 
            0 10px 30px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.1);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(64, 224, 208, 0.2);
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .space-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 0deg, transparent, rgba(64, 224, 208, 0.1), transparent);
        animation: rotate 10s linear infinite;
        z-index: -1;
    }

    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }

    .space-header h1 {
        font-family: 'Orbitron', monospace !important;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin: 0 0 1rem 0 !important;
        background: linear-gradient(135deg, #40e0d0 0%, #48cae4 50%, #7209b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(64, 224, 208, 0.5);
        animation: glow 3s ease-in-out infinite alternate;
        text-align: center;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(64, 224, 208, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(64, 224, 208, 0.8)); }
    }

    .space-header h3 {
        font-family: 'Exo 2', sans-serif !important;
        color: #b8c5d1 !important;
        font-size: 1.2rem !important;
        font-weight: 300 !important;
        margin: 0 0 1.5rem 0 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        text-align: center;
    }

    /* Chat container */
    .chat-container {
        background: linear-gradient(145deg, rgba(15,15,35,0.9) 0%, rgba(25,25,45,0.9) 100%) !important;
        border: 2px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 20px !important;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1) !important;
        height: 500px !important;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        width: 100% !important;
    }

    .chat-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(64, 224, 208, 0.8), transparent);
        animation: scan 3s ease-in-out infinite;
    }

    @keyframes scan {
        0%, 100% { transform: translateX(-100%); opacity: 0; }
        50% { transform: translateX(100%); opacity: 1; }
    }

    /* Input and button area */
    .input-section {
        display: flex !important;
        flex-direction: column !important;
        gap: 15px !important;
        width: 100% !important;
    }

    .input-area {
        background: linear-gradient(145deg, rgba(15,15,35,0.9) 0%, rgba(25,25,45,0.9) 100%) !important;
        border: 2px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 15px !important;
        color: #e0e6ed !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .input-area:focus-within {
        border-color: rgba(64, 224, 208, 0.8) !important;
        box-shadow: 
            0 0 20px rgba(64, 224, 208, 0.3),
            0 4px 16px rgba(0,0,0,0.2) !important;
    }

    /* Button row */
    .button-row {
        display: flex !important;
        gap: 15px !important;
        width: 100% !important;
        justify-content: flex-start !important;
    }

    /* Buttons */
    .btn-primary {
        background: linear-gradient(135deg, #40e0d0 0%, #48cae4 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #0a0a15 !important;
        font-weight: 700 !important;
        font-family: 'Orbitron', monospace !important;
        padding: 14px 28px !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 
            0 4px 15px rgba(64, 224, 208, 0.4),
            inset 0 1px 0 rgba(255,255,255,0.2) !important;
        position: relative;
        overflow: hidden;
        flex: 1 !important;
        max-width: 200px !important;
    }

    .btn-primary::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }

    .btn-primary:hover::before {
        left: 100%;
    }

    .btn-primary:hover {
        background: linear-gradient(135deg, #48cae4 0%, #7209b7 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 
            0 8px 25px rgba(64, 224, 208, 0.6),
            inset 0 1px 0 rgba(255,255,255,0.2) !important;
    }

    .btn-secondary {
        background: linear-gradient(135deg, rgba(15,15,35,0.9) 0%, rgba(25,25,45,0.9) 100%) !important;
        border: 2px solid rgba(64, 224, 208, 0.5) !important;
        border-radius: 12px !important;
        color: #40e0d0 !important;
        font-weight: 600 !important;
        font-family: 'Orbitron', monospace !important;
        padding: 14px 28px !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease !important;
        flex: 1 !important;
        max-width: 150px !important;
    }

    .btn-secondary:hover {
        background: linear-gradient(135deg, rgba(64, 224, 208, 0.2) 0%, rgba(72, 202, 228, 0.2) 100%) !important;
        border-color: rgba(64, 224, 208, 0.8) !important;
        color: #fff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(64, 224, 208, 0.3) !important;
    }

    /* Side panels */
    .space-panel {
        background: linear-gradient(145deg, rgba(15,15,35,0.9) 0%, rgba(25,25,45,0.9) 100%) !important;
        border: 2px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 20px !important;
        padding: 20px !important;
        backdrop-filter: blur(10px);
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1) !important;
        position: relative;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .space-panel h3 {
        font-family: 'Orbitron', monospace !important;
        color: #40e0d0 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin: 0 0 15px 0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Upload area */
    .upload-area {
        background: linear-gradient(145deg, rgba(64, 224, 208, 0.1) 0%, rgba(72, 202, 228, 0.1) 100%) !important;
        border: 3px dashed rgba(64, 224, 208, 0.5) !important;
        border-radius: 15px !important;
        padding: 30px 20px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(5px);
        min-height: 100px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .upload-area:hover {
        background: linear-gradient(145deg, rgba(64, 224, 208, 0.2) 0%, rgba(72, 202, 228, 0.2) 100%) !important;
        border-color: rgba(64, 224, 208, 0.8) !important;
        box-shadow: 0 0 30px rgba(64, 224, 208, 0.3) !important;
    }

    /* Status box */
    .status-box {
        background: linear-gradient(145deg, rgba(64, 224, 208, 0.1) 0%, rgba(72, 202, 228, 0.1) 100%) !important;
        border-left: 4px solid #40e0d0 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        font-family: 'Exo 2', monospace !important;
        font-size: 13px !important;
        color: #b8c5d1 !important;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* System info */
    .system-info {
        background: linear-gradient(145deg, rgba(15,15,35,0.95) 0%, rgba(25,25,45,0.95) 100%) !important;
        color: #b8c5d1 !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        font-family: 'Exo 2', monospace !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    /* Footer */
    .space-footer {
        background: linear-gradient(135deg, rgba(15,15,35,0.9) 0%, rgba(25,25,45,0.9) 100%) !important;
        border-top: 2px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 30px 30px 0 0 !important;
        padding: 30px 20px !important;
        margin: 20px 0 0 0 !important;
        text-align: center !important;
        color: #b8c5d1 !important;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
        width: 100% !important;
        box-sizing: border-box !important;
    }

    /* Accordions */
    .gr-accordion {
        background: rgba(15,15,35,0.7) !important;
        border: 1px solid rgba(64, 224, 208, 0.3) !important;
        border-radius: 15px !important;
        backdrop-filter: blur(10px);
        margin-bottom: 15px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .gr-accordion summary {
        background: rgba(25,25,45,0.8) !important;
        color: #40e0d0 !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 600 !important;
        padding: 15px !important;
        border-radius: 15px 15px 0 0 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15,15,35,0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #40e0d0 0%, #48cae4 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #48cae4 0%, #7209b7 100%);
    }

    /* Gradio overrides */
    .gr-row {
        gap: 0px !important;
    }
    
    .gr-column {
        min-width: 0 !important;
    }
    """

    # Create the space-themed interface with improved layout
    with gr.Blocks(
        title="🚀 AI Research Assistant - Space Command",
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Exo 2")
        ),
        css=css
    ) as app:

        # Space-themed header
        with gr.Row(elem_classes=["space-header"]):
            gr.HTML("""
                <div style="text-align: center; position: relative; z-index: 1;">
                    <h1>🚀 NEXUS AI COMMAND</h1>
                    <h3>◆ Advanced Neural Research Interface ◆</h3>
                    <p style="margin: 0; color: #8892b0; font-size: 16px; font-weight: 300;">
                        Quantum-powered document analysis and real-time cosmic data retrieval
                    </p>
                    <div style="margin-top: 1rem; font-family: 'Orbitron', monospace; font-size: 12px; color: #40e0d0; letter-spacing: 2px;">
                        [ SYSTEM OPERATIONAL - ALL NETWORKS ONLINE ]
                    </div>
                </div>
            """)

        # Main content layout with improved alignment
        with gr.Row(elem_classes=["main-content-row"]):
            # Left column - Chat interface
            with gr.Column(elem_classes=["left-content"]):
                # Chat conversation area
                chatbot = gr.Chatbot(
                    label="🛸 Neural Interface",
                    height=500,
                    show_copy_button=True,
                    render_markdown=True,
                    elem_classes=["chat-container"],
                    avatar_images=("👤", "🤖"),
                    type="messages",
                    placeholder="Initializing quantum communication channel..."
                )
                
                # Input section
                with gr.Column(elem_classes=["input-section"]):
                    msg = gr.Textbox(
                        placeholder="⚡ Transmit your query to the AI neural network...",
                        label="Command Input",
                        lines=3,
                        container=True,
                        elem_classes=["input-area"]
                    )
                    
                    # Action buttons
                    with gr.Row(elem_classes=["button-row"]):
                        submit_btn = gr.Button(
                            "🚀 TRANSMIT",
                            variant="primary",
                            elem_classes=["btn-primary"]
                        )
                        clear_btn = gr.Button(
                            "🗑️ RESET",
                            elem_classes=["btn-secondary"]
                        )

            # Right column - Control panels
            with gr.Column(elem_classes=["right-content"]):
                # Document upload section
                with gr.Group(elem_classes=["space-panel"]):
                    gr.HTML("<h3>📡 Document Scanner</h3>")
                    
                    files = gr.File(
                        label="Upload Neural Data Files",
                        file_count="multiple",
                        file_types=[".pdf"],
                        elem_classes=["upload-area"],
                        interactive=True
                    )
                    
                    gr.HTML("""
                        <div style="margin-top: 1rem; padding: 1rem; background: rgba(64, 224, 208, 0.05); border-radius: 8px; border-left: 3px solid #40e0d0;">
                            <div style="color: #40e0d0; font-weight: 600; font-size: 12px; margin-bottom: 0.5rem;">◆ SCAN PARAMETERS</div>
                            <div style="font-size: 11px; color: #b8c5d1; line-height: 1.4;">
                                • Multi-file quantum processing<br>
                                • Maximum payload: 50MB per file<br>
                                • Processing time: 10-30 seconds<br>
                                • Secure neural encryption active
                            </div>
                        </div>
                    """)

                # Status monitoring
                with gr.Group(elem_classes=["space-panel"]):
                    gr.HTML("<h3>⚡ System Status</h3>")
                    
                    status = gr.Textbox(
                        label="Mission Control",
                        value="🛸 Systems Online",
                        interactive=False,
                        container=True,
                        elem_classes=["status-box"]
                    )

                # System information
                with gr.Accordion("🔧 Technical Readout", open=False):
                    system_info = gr.HTML(
                        get_system_info(),
                        elem_classes=["system-info"]
                    )
                    
                    refresh_btn = gr.Button(
                        "🔄 UPDATE",
                        size="sm",
                        elem_classes=["btn-secondary"]
                    )

                # Command examples
                with gr.Accordion("⚡ Command Library", open=False):
                    gr.HTML("""
                        <div style="padding: 1rem;">
                            <h4 style="color: #40e0d0; font-family: 'Orbitron', monospace; margin-bottom: 1rem; font-size: 11px;">📡 DATA ANALYSIS</h4>
                            <div style="font-size: 11px; color: #b8c5d1; margin-bottom: 1rem; font-family: 'Exo 2', monospace;">
                                • "Analyze neural patterns in data"<br>
                                • "Extract quantum signatures"<br>
                                • "Perform deep methodology scan"
                            </div>
                            
                            <h4 style="color: #40e0d0; font-family: 'Orbitron', monospace; margin-bottom: 1rem; font-size: 11px;">🌌 COSMIC INTELLIGENCE</h4>
                            <div style="font-size: 11px; color: #b8c5d1; margin-bottom: 1rem; font-family: 'Exo 2', monospace;">
                                • "Access starfleet intelligence"<br>
                                • "Scan tech sector anomalies"<br>
                                • "Retrieve quantum research"
                            </div>
                        </div>
                    """)

        # Space-themed footer
        with gr.Row(elem_classes=["space-footer"]):
            gr.HTML("""
                <div>
                    <div style="font-family: 'Orbitron', monospace; color: #40e0d0; font-weight: 700; font-size: 16px; margin-bottom: 1rem;">
                        ◆ SECURE NEURAL NETWORK ◆
                    </div>
                    <p style="margin: 0; font-size: 13px; color: #8892b0; font-family: 'Exo 2', sans-serif;">
                        Quantum encryption active • Neural processing secured • Zero data persistence<br>
                        <span style="color: #40e0d0;">POWERED BY:</span> Ollama Neural Core • LangChain Quantum Agents • ChromaDB Matrix • Semantic Vector Analysis
                    </p>
                </div>
            """)

        # Event handlers (keep existing functionality)
        def handle_message(message, history, files):
            return chat_function(message, history, files)

        def handle_clear():
            return clear_chat()

        def handle_refresh():
            return get_system_info()

        # Wire up interactions
        submit_btn.click(
            fn=handle_message,
            inputs=[msg, chatbot, files],
            outputs=[msg, chatbot, status, system_info]
        )

        msg.submit(
            fn=handle_message,
            inputs=[msg, chatbot, files],
            outputs=[msg, chatbot, status, system_info]
        )

        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, status, system_info]
        )

        refresh_btn.click(
            fn=handle_refresh,
            outputs=[system_info]
        )

        # Initialize on load
        def startup():
            global assistant
            try:
                assistant = initialize_assistant()
                if assistant:
                    return "🛸 Neural Networks Online", get_system_info()
                else:
                    return "⚠️ Initialization Failed", get_system_info()
            except Exception as e:
                logger.error(f"Startup error: {str(e)}")
                return f"⚠️ System Error: {str(e)}", get_system_info()

        app.load(
            fn=startup,
            outputs=[status, system_info]
        )

        return app


def main():
    try:
        logger.info("Starting AI Research Assistant Web Application...")
        app = create_interface()
        app.queue(max_size=20)

        server_name = "0.0.0.0"
        server_port = int(getattr(Config, "GRADIO_PORT", 7860))
        share = bool(getattr(Config, "GRADIO_SHARE", False))

        if os.getenv("SPACE_ID"):
            logger.info("Detected Hugging Face Spaces environment")
            server_name = "0.0.0.0"
            server_port = 7860
            share = False

        if os.getenv("GRADIO_DEV"):
            logger.info("Development mode detected (GRADIO_DEV)")
            share = True

        logger.info(f"Launching Gradio app on {server_name}:{server_port}")
        logger.info(f"Configuration: Share={share}, Queue=enabled")

        app.launch(server_name=server_name, server_port=server_port, share=share, show_error=True)

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        logger.error(traceback.format_exc())
        print("\nApplication Startup Failed")
        print(f"Error: {str(e)}")
        print("\nTroubleshooting Steps:")
        print("1. pip install -r requirements.txt")
        print("2. Verify .env has TAVILY_API_KEY if using web search")
        print("3. Run from project root")
        print("4. Check the stack trace above")
        raise

if __name__ == "__main__":
    main()
