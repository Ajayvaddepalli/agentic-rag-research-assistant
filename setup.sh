#!/bin/bash

# AI Research Assistant - Quick Setup Script
# This script automates the complete setup process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Main setup function
main() {
    echo -e "${BLUE}"
    echo "🚀 AI Research Assistant - Quick Setup"
    echo "======================================"
    echo -e "${NC}"
    
    # Step 1: Check Prerequisites
    print_status "Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            print_error "Python is not installed. Please install Python 3.10+ first."
            exit 1
        fi
        PYTHON_CMD="python"
    else
        PYTHON_CMD="python3"
    fi
    
    # Verify Python version
    PYTHON_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Found Python ${PYTHON_VERSION}"
    
    if [[ "$(printf '%s\n' "3.10" "${PYTHON_VERSION}" | sort -V | head -n1)" != "3.10" ]]; then
        print_warning "Python 3.10+ is recommended. Current version: ${PYTHON_VERSION}"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_warning "Git is not installed. Version control features will be limited."
    else
        print_success "Git found"
    fi
    
    # Step 2: Create Virtual Environment
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    ${PYTHON_CMD} -m venv venv
    print_success "Virtual environment created"
    
    # Step 3: Activate Virtual Environment
    print_status "Activating virtual environment..."
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        # Windows
        source venv/Scripts/activate
        print_status "Activated virtual environment (Windows)"
    else
        # Linux/Mac
        source venv/bin/activate
        print_status "Activated virtual environment (Unix)"
    fi
    
    # Step 4: Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
    
    # Step 5: Install Dependencies
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Step 6: Set up Environment Configuration
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Created .env file from template"
            print_warning "⚠️  Please edit .env file and add your API keys!"
        else
            print_error ".env.example template not found!"
            exit 1
        fi
    else
        print_warning ".env file already exists. Skipping..."
    fi
    
    # Step 7: Create necessary directories
    print_status "Creating necessary directories..."
    
    mkdir -p data/sample_pdfs
    mkdir -p data/vector_db
    mkdir -p logs
    
    print_success "Directories created"
    
    # Step 8: Run Basic Tests
    print_status "Running basic functionality tests..."
    
    if [ -f "tests/test_suite.py" ]; then
        if ${PYTHON_CMD} tests/test_suite.py --basic; then
            print_success "Basic tests passed!"
        else
            print_warning "Some tests failed, but setup can continue"
        fi
    else
        print_warning "Test suite not found, skipping tests"
    fi
    
    # Step 9: Display Setup Summary
    echo ""
    echo -e "${GREEN}🎉 Setup Complete!${NC}"
    echo "=================="
    echo ""
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo "1. Edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY=your_openai_key_here"
    echo "   - TAVILY_API_KEY=your_tavily_key_here"
    echo ""
    echo "2. Activate the virtual environment:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "   source venv/Scripts/activate"
    else
        echo "   source venv/bin/activate"
    fi
    echo ""
    echo "3. Start the application:"
    echo "   python app/app.py"
    echo ""
    echo "4. Open your browser and go to:"
    echo "   http://localhost:7860"
    echo ""
    echo -e "${BLUE}🔑 Get API Keys:${NC}"
    echo "- OpenAI: https://platform.openai.com/api-keys"
    echo "- Tavily: https://tavily.com (free tier: 1000 searches/month)"
    echo ""
    echo -e "${BLUE}📚 Documentation:${NC}"
    echo "- README.md - Complete project documentation"
    echo "- DEPLOYMENT_GUIDE.md - Deployment instructions"
    echo "- TESTING_GUIDE.md - Testing procedures"
    echo ""
    echo -e "${BLUE}🆘 Need Help?${NC}"
    echo "- Run tests: python tests/test_suite.py --basic"
    echo "- Check logs: tail -f logs/app.log"
    echo "- GitHub Issues: https://github.com/yourusername/ai-research-assistant/issues"
    
    # Final status check
    echo ""
    print_status "Setup Summary:"
    echo "✅ Python environment: Ready"
    echo "✅ Dependencies: Installed"
    echo "✅ Configuration: Template created"
    echo "✅ Directories: Created"
    
    if [ -f ".env" ]; then
        # Check if API keys are configured
        if grep -q "your_.*_key_here" .env; then
            echo "⚠️  API Keys: Need configuration"
            print_warning "Remember to add your API keys to .env file!"
        else
            echo "✅ API Keys: Configured"
        fi
    fi
    
    echo ""
    print_success "AI Research Assistant is ready to use! 🚀"
}

# Function to show help
show_help() {
    echo "AI Research Assistant - Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  --skip-tests   Skip running basic tests"
    echo "  --clean        Clean previous installation before setup"
    echo ""
    echo "Examples:"
    echo "  $0              # Normal setup"
    echo "  $0 --clean      # Clean setup"
    echo "  $0 --skip-tests # Setup without running tests"
}

# Function to clean previous installation
clean_installation() {
    print_status "Cleaning previous installation..."
    
    if [ -d "venv" ]; then
        rm -rf venv
        print_status "Removed virtual environment"
    fi
    
    if [ -d "data/vector_db" ]; then
        rm -rf data/vector_db
        print_status "Cleared vector database"
    fi
    
    if [ -d "__pycache__" ]; then
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        print_status "Cleared Python cache"
    fi
    
    if [ -f "app.log" ]; then
        rm -f app.log
        print_status "Cleared log files"
    fi
    
    print_success "Cleanup complete"
}

# Parse command line arguments
SKIP_TESTS=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute main function
if [ "$CLEAN" = true ]; then
    clean_installation
fi

main

# Export setup completion status
export AI_RESEARCH_ASSISTANT_SETUP_COMPLETE=true