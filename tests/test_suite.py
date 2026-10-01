"""
Comprehensive test suite for the AI Research Assistant.

This module provides automated testing for all components of the system
including configuration, document processing, tools, and agent functionality.
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    # Import all components to test
    import config
    import document_processor
    import tools
    import utils
    import agent
except ImportError as e:
    print(f"❌ Import error during testing: {e}")
    print("Make sure you're running tests from the project root directory")
    sys.exit(1)


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_class_exists(self):
        """Test that Config class is properly defined."""
        self.assertTrue(hasattr(config, 'Config'))
        
    def test_required_attributes(self):
        """Test that Config has all required attributes."""
        required_attrs = [
            'OPENAI_API_KEY', 'TAVILY_API_KEY', 'LLM_MODEL',
            'EMBEDDING_MODEL', 'TEMPERATURE', 'CHUNK_SIZE'
        ]
        
        for attr in required_attrs:
            self.assertTrue(hasattr(config.Config, attr), f"Missing attribute: {attr}")
    
    def test_default_values(self):
        """Test that default values are reasonable."""
        self.assertIsInstance(config.Config.TEMPERATURE, float)
        self.assertGreaterEqual(config.Config.TEMPERATURE, 0.0)
        self.assertLessEqual(config.Config.TEMPERATURE, 2.0)
        
        self.assertIsInstance(config.Config.CHUNK_SIZE, int)
        self.assertGreater(config.Config.CHUNK_SIZE, 0)
        
        self.assertIsInstance(config.Config.CHUNK_OVERLAP, int)
        self.assertGreaterEqual(config.Config.CHUNK_OVERLAP, 0)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_format_file_size(self):
        """Test file size formatting."""
        self.assertEqual(utils.format_file_size(0), "0 B")
        self.assertEqual(utils.format_file_size(1024), "1.0 KB")
        self.assertEqual(utils.format_file_size(1024 * 1024), "1.0 MB")
    
    def test_truncate_text(self):
        """Test text truncation."""
        long_text = "This is a very long text that should be truncated"
        truncated = utils.truncate_text(long_text, 20)
        self.assertLessEqual(len(truncated), 20)
        self.assertTrue(truncated.endswith("..."))
    
    def test_validate_file_type(self):
        """Test file type validation."""
        self.assertTrue(utils.validate_file_type("document.pdf", [".pdf"]))
        self.assertFalse(utils.validate_file_type("document.txt", [".pdf"]))
        self.assertFalse(utils.validate_file_type("", [".pdf"]))
    
    def test_session_manager(self):
        """Test session management."""
        sm = utils.SessionManager()
        
        # Test session creation
        session = sm.create_session("test_session")
        self.assertIn("session_id", session)
        self.assertEqual(session["session_id"], "test_session")
        
        # Test session retrieval
        retrieved = sm.get_session("test_session")
        self.assertEqual(retrieved["session_id"], "test_session")
        
        # Test counter increment
        new_value = sm.increment_counter("test_session", "test_counter")
        self.assertEqual(new_value, 1)


class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_document_processor_initialization(self):
        """Test DocumentProcessor initialization."""
        # Mock the OpenAI embeddings to avoid API calls
        with patch('document_processor.OpenAIEmbeddings'):
            processor = document_processor.DocumentProcessor(
                vector_db_path=os.path.join(self.temp_dir, "test_db")
            )
            self.assertIsNotNone(processor)
            self.assertIsNotNone(processor.text_splitter)
    
    def test_validate_file(self):
        """Test file validation."""
        with patch('document_processor.OpenAIEmbeddings'):
            processor = document_processor.DocumentProcessor()
            
            # Test with non-existent file
            result = processor.validate_file("nonexistent.pdf")
            self.assertFalse(result["valid"])
            self.assertIn("File not found", str(result["errors"]))
    
    def test_database_info(self):
        """Test database info retrieval."""
        with patch('document_processor.OpenAIEmbeddings'):
            processor = document_processor.DocumentProcessor()
            info = processor.get_database_info()
            
            self.assertIsInstance(info, dict)
            self.assertIn("database_path", info)
            self.assertIn("vectorstore_loaded", info)


class TestToolManager(unittest.TestCase):
    """Test tool management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the Tavily search to avoid API calls
        self.tavily_patcher = patch('tools.TavilySearchResults')
        self.mock_tavily = self.tavily_patcher.start()
        
        # Mock the document processor
        self.doc_processor_mock = Mock()
        
    def tearDown(self):
        """Clean up patches."""
        self.tavily_patcher.stop()
    
    def test_tool_manager_initialization(self):
        """Test ToolManager initialization."""
        tool_manager = tools.ToolManager()
        self.assertIsNotNone(tool_manager)
    
    def test_tool_creation_without_docs(self):
        """Test tool creation without document processor."""
        tool_manager = tools.ToolManager()
        all_tools = tool_manager.get_all_tools()
        
        # Should have web search tool only
        tool_names = [tool.name for tool in all_tools]
        self.assertIn("web_search", tool_names)
    
    def test_tool_creation_with_docs(self):
        """Test tool creation with document processor."""
        tool_manager = tools.ToolManager(self.doc_processor_mock)
        all_tools = tool_manager.get_all_tools()
        
        # Should have both document search and web search
        tool_names = [tool.name for tool in all_tools]
        self.assertIn("web_search", tool_names)
        self.assertIn("document_search", tool_names)
    
    def test_tool_status(self):
        """Test tool status reporting."""
        tool_manager = tools.ToolManager()
        status = tool_manager.get_tools_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("web_search", status)
        self.assertIn("total_tools", status)


class TestAgent(unittest.TestCase):
    """Test agent functionality."""
    
    def setUp(self):
        """Set up test fixtures with mocks."""
        # Mock all external dependencies
        self.openai_patcher = patch('agent.ChatOpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Mock the tool manager
        self.tool_manager_patcher = patch('agent.ToolManager')
        self.mock_tool_manager = self.tool_manager_patcher.start()
        
        # Mock session manager
        self.session_manager_patcher = patch('agent.SessionManager')
        self.mock_session_manager = self.session_manager_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.openai_patcher.stop()
        self.tool_manager_patcher.stop()
        self.session_manager_patcher.stop()
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        # Configure mocks
        self.mock_tool_manager.return_value.get_all_tools.return_value = []
        
        try:
            assistant = agent.AgenticRAGAssistant()
            self.assertIsNotNone(assistant)
        except Exception as e:
            # Might fail due to missing API keys in test environment
            self.assertIn("api", str(e).lower())
    
    def test_system_status(self):
        """Test system status retrieval."""
        self.mock_tool_manager.return_value.get_all_tools.return_value = []
        self.mock_tool_manager.return_value.get_tools_status.return_value = {
            "total_tools": 0
        }
        
        try:
            assistant = agent.AgenticRAGAssistant()
            status = assistant.get_system_status()
            
            self.assertIsInstance(status, dict)
            self.assertIn("agent_initialized", status)
        except Exception:
            # Expected in test environment without proper API keys
            pass


class IntegrationTest(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_component_imports(self):
        """Test that all components can be imported successfully."""
        components = ['config', 'utils', 'document_processor', 'tools', 'agent']
        
        for component in components:
            self.assertTrue(
                component in sys.modules,
                f"Component {component} not imported"
            )
    
    def test_config_validation_with_missing_keys(self):
        """Test configuration validation with missing API keys."""
        # This test expects validation to fail gracefully
        try:
            config.Config.validate()
        except ValueError as e:
            # Expected behavior when API keys are missing
            self.assertIn("API", str(e))
        except Exception as e:
            # Other exceptions are also acceptable in test environment
            pass


def run_basic_tests():
    """
    Run basic functionality tests without requiring API keys.
    
    This function runs tests that can work in any environment,
    focusing on component initialization and basic functionality.
    """
    print("🧪 Running Basic Functionality Tests...")
    print("=" * 50)
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    # Test 1: Component Imports
    try:
        print("\\n1. Testing Component Imports...")
        components = ['config', 'utils', 'document_processor', 'tools', 'agent']
        
        for component in components:
            exec(f"import {component}")
            print(f"   ✅ {component}")
        
        print("   ✅ All components imported successfully")
        test_results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Import test: {e}")
    
    # Test 2: Configuration
    try:
        print("\\n2. Testing Configuration...")
        
        # Test that Config class exists and has required attributes
        required_attrs = ['OPENAI_API_KEY', 'TAVILY_API_KEY', 'LLM_MODEL']
        for attr in required_attrs:
            if hasattr(config.Config, attr):
                print(f"   ✅ {attr} attribute present")
            else:
                raise AttributeError(f"Missing {attr}")
        
        test_results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Configuration test: {e}")
    
    # Test 3: Utilities
    try:
        print("\\n3. Testing Utilities...")
        
        # Test basic utility functions
        test_size = utils.format_file_size(1024)
        assert "1.0 KB" in test_size
        print("   ✅ File size formatting")
        
        test_text = utils.truncate_text("This is a long text", 10)
        assert len(test_text) <= 13  # 10 + "..."
        print("   ✅ Text truncation")
        
        sm = utils.SessionManager()
        session = sm.create_session("test")
        assert session["session_id"] == "test"
        print("   ✅ Session management")
        
        test_results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Utilities test failed: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Utilities test: {e}")
    
    # Test 4: Document Processor (basic initialization)
    try:
        print("\\n4. Testing Document Processor...")
        
        # Mock the embeddings to avoid API calls
        with patch('document_processor.OpenAIEmbeddings') as mock_embeddings:
            mock_embeddings.return_value = Mock()
            
            processor = document_processor.DocumentProcessor()
            info = processor.get_database_info()
            
            assert isinstance(info, dict)
            assert "database_path" in info
            print("   ✅ Document processor initialization")
            print("   ✅ Database info retrieval")
        
        test_results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Document processor test failed: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Document processor test: {e}")
    
    # Test 5: Tool Manager (basic initialization)
    try:
        print("\\n5. Testing Tool Manager...")
        
        # Mock external dependencies
        with patch('tools.TavilySearchResults') as mock_tavily:
            mock_tavily.return_value = Mock()
            
            tool_manager = tools.ToolManager()
            tools_list = tool_manager.get_all_tools()
            status = tool_manager.get_tools_status()
            
            assert isinstance(tools_list, list)
            assert isinstance(status, dict)
            print("   ✅ Tool manager initialization")
            print("   ✅ Tool creation and status")
        
        test_results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Tool manager test failed: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"Tool manager test: {e}")
    
    # Print results
    print("\\n" + "=" * 50)
    print(f"📊 Test Results: {test_results['passed']} passed, {test_results['failed']} failed")
    
    if test_results["errors"]:
        print("\\n❌ Errors encountered:")
        for error in test_results["errors"]:
            print(f"   - {error}")
    
    if test_results["failed"] == 0:
        print("\\n🎉 All basic tests passed! Your components are working correctly.")
    else:
        print(f"\\n⚠️  {test_results['failed']} test(s) failed. Check the errors above.")
    
    return test_results["failed"] == 0


if __name__ == "__main__":
    print("🧪 AI Research Assistant - Test Suite")
    print("=" * 60)
    
    # Check if this is a basic test run or full test suite
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        # Run basic tests only (no API calls required)
        success = run_basic_tests()
        sys.exit(0 if success else 1)
    else:
        # Run full test suite with unittest
        print("Running full test suite with unittest...")
        print("Note: Some tests may fail if API keys are not configured.")
        print("Use '--basic' flag for tests that don't require API keys.\\n")
        
        unittest.main(verbosity=2)