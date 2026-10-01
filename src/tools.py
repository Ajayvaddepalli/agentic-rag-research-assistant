"""
Tools for the AI Research Assistant agent.

This module defines the tools that the agent can use:
- Document retrieval tool for searching uploaded PDFs
- Web search tool for real-time information
"""

import logging
from typing import List, Optional, Dict, Any
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

from .document_processor import DocumentProcessor
from .config import Config


class ToolManager:
    """
    Manages all tools available to the AI agent.
    
    This class is responsible for creating and managing the tools that
    the agent can use to answer user queries. It handles both document
    search and web search capabilities.
    """
    
    def __init__(self, document_processor: Optional[DocumentProcessor] = None):
        """
        Initialize the tool manager.
        
        Args:
            document_processor: Optional DocumentProcessor instance for document search
        """
        self.document_processor = document_processor
        self.logger = logging.getLogger(__name__)
        
        # Initialize web search tool
        try:
            self.web_search_tool = TavilySearchResults(
                max_results=Config.WEB_SEARCH_RESULTS,
                search_depth="advanced",  # More comprehensive search
                include_answer=True,      # Include AI-generated answer
                include_raw_content=False,  # Don't include full HTML
                include_images=False,     # Don't include images for faster processing
                api_key=Config.TAVILY_API_KEY
            )
            self.logger.info("Web search tool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize web search tool: {e}")
            self.web_search_tool = None
    
    def create_document_retrieval_tool(self) -> Tool:
        """
        Create a tool for retrieving information from uploaded documents.
        
        Returns:
            LangChain Tool for document retrieval
        """
        def search_documents(query: str) -> str:
            """
            Search through uploaded documents for relevant information.
            
            This function searches the vector database of uploaded documents
            and returns the most relevant excerpts that match the query.
            
            Args:
                query: The search query string
                
            Returns:
                Formatted string with relevant document excerpts and metadata
            """
            try:
                # Check if document processor is available
                if self.document_processor is None:
                    return ("❌ **No documents available**: No documents have been uploaded yet. "
                           "Please upload PDF documents first to search through them.")
                
                # Get retriever
                retriever = self.document_processor.get_retriever()
                if retriever is None:
                    return ("❌ **No documents processed**: No documents are available for search. "
                           "Please upload and process PDF documents first.")
                
                # Perform document search
                self.logger.info(f"Searching documents for query: '{query[:50]}...'")
                docs = self.document_processor.search_documents(
                    query, 
                    k=Config.DOCUMENT_SEARCH_RESULTS
                )
                
                if not docs:
                    return (f"🔍 **No relevant information found**: I couldn't find any relevant "
                           f"information in the uploaded documents for the query: '{query}'. "
                           f"You might want to try rephrasing your question or check if the "
                           f"information exists in the uploaded documents.")
                
                # Format results with enhanced metadata
                results = []
                results.append(f"📄 **Found {len(docs)} relevant sections in uploaded documents:**\\n")
                
                for i, doc in enumerate(docs, 1):
                    content = doc.page_content.strip()
                    
                    # Extract metadata
                    file_name = doc.metadata.get('file_name', 'Unknown document')
                    page_num = doc.metadata.get('page', doc.metadata.get('page_number', 'Unknown'))
                    chunk_id = doc.metadata.get('chunk_id', i)
                    
                    # Create a clean, readable format
                    result_header = f"**📋 Result {i}** ({file_name}"
                    if page_num != 'Unknown':
                        result_header += f", Page {page_num}"
                    result_header += f"):"
                    
                    # Truncate very long content for readability
                    if len(content) > 800:
                        content = content[:800] + "... [content truncated]"
                    
                    results.append(f"{result_header}\\n{content}")
                
                # Add helpful footer
                results.append(f"\\n💡 **Tip**: These results are from your uploaded documents. "
                              f"Ask follow-up questions to get more specific information!")
                
                formatted_result = "\\n\\n".join(results)
                self.logger.info(f"Document search completed: returned {len(docs)} results")
                return formatted_result
                
            except Exception as e:
                error_msg = f"❌ **Search error**: An error occurred while searching documents: {str(e)}"
                self.logger.error(f"Document search error: {str(e)}")
                return error_msg
        
        return Tool(
            name="document_search",
            description="""
            **Search uploaded PDF documents for specific information.**
            
            Use this tool when the user asks about:
            - Content from their uploaded PDF documents
            - Specific facts, quotes, or data from the documents
            - Analysis of information in the uploaded files
            - Details about research papers, reports, or manuals they've shared
            - Comparisons or summaries of document content
            
            This tool searches through the semantic content of uploaded documents
            and returns the most relevant excerpts with source information.
            
            **When to use**: 
            - User asks "What does the document say about..."
            - User requests summaries of uploaded content
            - User asks for specific data from their files
            - User wants quotes or references from documents
            
            **Input**: A clear, specific search query related to document content.
            **Output**: Relevant excerpts from documents with source attribution.
            """.strip(),
            func=search_documents
        )
    
    def create_web_search_tool(self) -> Tool:
        """
        Create a tool for searching the web for current information.
        
        Returns:
            LangChain Tool for web search
        """
        def search_web(query: str) -> str:
            """
            Search the web for current information, news, and general knowledge.
            
            This function uses the Tavily search API to find relevant, current
            information from the web and returns formatted results.
            
            Args:
                query: The search query string
                
            Returns:
                Formatted string with web search results
            """
            try:
                # Check if web search tool is available
                if self.web_search_tool is None:
                    return ("❌ **Web search unavailable**: Web search functionality is not available. "
                           "Please check the Tavily API key configuration.")
                
                self.logger.info(f"Performing web search for query: '{query[:50]}...'")
                
                # Perform web search
                search_results = self.web_search_tool.run(query)
                
                # Handle different response formats
                if isinstance(search_results, str):
                    # If it's already a formatted string, return it
                    return f"🌐 **Web Search Results for '{query}':**\\n\\n{search_results}"
                
                elif isinstance(search_results, list) and search_results:
                    # Format list of results
                    results = []
                    results.append(f"🌐 **Found {len(search_results)} web results for '{query}':**\\n")
                    
                    for i, result in enumerate(search_results, 1):
                        if isinstance(result, dict):
                            title = result.get('title', 'No title available')
                            content = result.get('content', result.get('snippet', 'No content available'))
                            url = result.get('url', 'No URL available')
                            
                            # Clean up title and content
                            title = title.strip()
                            content = content.strip()
                            
                            # Truncate very long content
                            if len(content) > 600:
                                content = content[:600] + "... [truncated]"
                            
                            result_text = f"**🔗 Result {i}: {title}**\\n{content}\\n*Source: {url}*"
                            results.append(result_text)
                        else:
                            # Handle string results
                            results.append(f"**🔗 Result {i}**: {str(result)}")
                    
                    # Add helpful footer
                    results.append(f"\\n💡 **Note**: These are current web results. "
                                  f"Information may be more recent than my training data.")
                    
                    formatted_result = "\\n\\n".join(results)
                    self.logger.info(f"Web search completed: returned {len(search_results)} results")
                    return formatted_result
                
                else:
                    # Handle empty or unexpected results
                    return (f"🔍 **No results found**: I couldn't find current web information "
                           f"for the query: '{query}'. You might want to try rephrasing your "
                           f"search or being more specific.")
                
            except Exception as e:
                error_msg = f"❌ **Web search error**: An error occurred while searching the web: {str(e)}"
                self.logger.error(f"Web search error: {str(e)}")
                return error_msg
        
        return Tool(
            name="web_search",
            description="""
            **Search the internet for current information and general knowledge.**
            
            Use this tool when the user asks about:
            - Current events, recent news, or breaking developments
            - Up-to-date facts, statistics, or data not in uploaded documents
            - General knowledge questions not covered in documents
            - Recent developments in any field (technology, science, politics, etc.)
            - Verification of information or fact-checking
            - Current trends, market conditions, or real-time data
            
            This tool searches the live web and returns current, relevant information
            with source attribution from authoritative websites.
            
            **When to use**:
            - User asks about "latest", "current", "recent", or "new" information
            - User asks about topics not covered in their documents
            - User wants to verify or update information from documents
            - User asks general knowledge questions
            - User requests current statistics or market data
            
            **Input**: A clear search query for current/general information.
            **Output**: Current web information with sources and links.
            """.strip(),
            func=search_web
        )
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all available tools for the agent.
        
        Returns:
            List of all configured and available tools
        """
        tools = []
        
        # Add document search tool if document processor is available
        if self.document_processor is not None:
            try:
                doc_tool = self.create_document_retrieval_tool()
                tools.append(doc_tool)
                self.logger.debug("Document search tool added")
            except Exception as e:
                self.logger.error(f"Failed to create document search tool: {e}")
        
        # Add web search tool if available
        if self.web_search_tool is not None:
            try:
                web_tool = self.create_web_search_tool()
                tools.append(web_tool)
                self.logger.debug("Web search tool added")
            except Exception as e:
                self.logger.error(f"Failed to create web search tool: {e}")
        
        self.logger.info(f"Created {len(tools)} tools for agent")
        return tools
    
    def update_document_processor(self, document_processor: DocumentProcessor) -> None:
        """
        Update the document processor instance.
        
        This method allows updating the document processor after new documents
        have been uploaded and processed.
        
        Args:
            document_processor: New DocumentProcessor instance
        """
        self.document_processor = document_processor
        self.logger.info("Document processor updated successfully")
    
    def get_tools_status(self) -> Dict[str, Any]:
        """
        Get the status of all tools.
        
        Returns:
            Dictionary with tool availability and status information
        """
        status = {
            "document_search": {
                "available": self.document_processor is not None,
                "ready": False,
                "error": None
            },
            "web_search": {
                "available": self.web_search_tool is not None,
                "ready": self.web_search_tool is not None,
                "error": None
            },
            "total_tools": 0
        }
        
        # Check document search readiness
        if self.document_processor is not None:
            try:
                retriever = self.document_processor.get_retriever()
                status["document_search"]["ready"] = retriever is not None
                if retriever is not None:
                    status["total_tools"] += 1
            except Exception as e:
                status["document_search"]["error"] = str(e)
        
        # Check web search readiness
        if self.web_search_tool is not None:
            status["total_tools"] += 1
        else:
            status["web_search"]["error"] = "Tavily API key not configured"
        
        return status
    
    def test_tools(self) -> Dict[str, Any]:
        """
        Test all available tools with simple queries.
        
        Returns:
            Dictionary with test results for each tool
        """
        test_results = {
            "document_search": {"tested": False, "success": False, "error": None},
            "web_search": {"tested": False, "success": False, "error": None}
        }
        
        # Test document search
        if self.document_processor is not None:
            try:
                doc_tool = self.create_document_retrieval_tool()
                result = doc_tool.func("test query")
                test_results["document_search"]["tested"] = True
                test_results["document_search"]["success"] = "error" not in result.lower()
                test_results["document_search"]["result_preview"] = result[:100] + "..."
            except Exception as e:
                test_results["document_search"]["tested"] = True
                test_results["document_search"]["error"] = str(e)
        
        # Test web search (with a simple, safe query)
        if self.web_search_tool is not None:
            try:
                web_tool = self.create_web_search_tool()
                result = web_tool.func("current date")
                test_results["web_search"]["tested"] = True
                test_results["web_search"]["success"] = "error" not in result.lower()
                test_results["web_search"]["result_preview"] = result[:100] + "..."
            except Exception as e:
                test_results["web_search"]["tested"] = True
                test_results["web_search"]["error"] = str(e)
        
        return test_results


def test_tool_manager():
    """Test the tool manager functionality."""
    print("🔧 Testing ToolManager...")
    
    try:
        # Test without document processor
        tool_manager = ToolManager()
        print("✅ ToolManager initialized without document processor")
        
        # Get tools
        tools = tool_manager.get_all_tools()
        print(f"✅ Created {len(tools)} tools")
        
        # Get status
        status = tool_manager.get_tools_status()
        print(f"✅ Tool status retrieved: {status['total_tools']} tools ready")
        
        # Print tool descriptions
        for tool in tools:
            print(f"📋 Tool '{tool.name}': {tool.description[:50]}...")
        
        print("🎉 ToolManager test completed successfully!")
        return tool_manager
        
    except Exception as e:
        print(f"❌ ToolManager test failed: {str(e)}")
        return None


if __name__ == "__main__":
    test_tool_manager()