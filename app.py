"""
Main entry point for Hugging Face Spaces deployment.
This file sets up the cloud deployment mode and runs the app.
"""
import os
import sys

# Force cloud deployment mode for HF Spaces
os.environ["DEPLOYMENT_MODE"] = "cloud"

# Add app directory to Python path
sys.path.append("./app")
sys.path.append(".")

try:
    # Import your main app from the app folder
    from app.app import main
    
    if __name__ == "__main__":
        print("🚀 Starting NEXUS AI Command on Hugging Face Spaces...")
        main()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)
