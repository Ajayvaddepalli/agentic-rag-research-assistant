"""
LangGraph-based Agentic RAG implementation for AI Research Assistant.

This module implements the core agent using LangGraph for state management
and intelligent tool orchestration.
"""

import os
import logging
from typing import TypedDict, List, Annotated, Sequence, Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from .config import Config
from .document_processor import DocumentProcessor
from .tools import ToolManager
from .utils import SessionManager

# Define the agent state schema
class AgentState(TypedDict):
    """
    State schema for the agent graph.
    This defines the structure of data that flows through the agent's
    reasoning process, including messages, context, and metadata.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents_uploaded: bool
    session_id: str
    user_intent: Optional[str]
    tool_results: Optional[Dict[str, Any]]

class AgenticRAGAssistant:
    """
    Advanced AI Research Assistant using Agentic RAG architecture.
    
    This class implements a sophisticated AI agent that can:
    - Intelligently choose between document search and web search
    - Maintain conversation context and memory
    - Provide well-researched, source-attributed responses
    - Handle complex multi-step queries
    - Work with multiple LLM providers (Ollama, Groq, OpenAI)
    """

    def __init__(self, document_processor: Optional[DocumentProcessor] = None):
        """
        Initialize the Agentic RAG Assistant.

        Args:
            document_processor: Optional DocumentProcessor for document search
        """
        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.document_processor = document_processor
        self.tool_manager = ToolManager(document_processor)
        self.session_manager = SessionManager()

        # Initialize the language model based on configuration
        self._initialize_llm()

        # Initialize memory for conversation persistence
        self.memory = MemorySaver()

        # Build the agent graph
        self.graph = None
        self._build_graph()

        self.logger.info("🚀 Agentic RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """Initialize the Language Model based on deployment mode."""
        try:
            llm_info = Config.get_llm_info()
            self.logger.info(f"🤖 Initializing {llm_info['provider']} LLM...")

            if Config.LLM_TYPE == "groq":
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model=Config.GROQ_MODEL,
                    temperature=Config.TEMPERATURE,
                    groq_api_key=Config.GROQ_API_KEY,
                    max_tokens=Config.MAX_TOKENS,
                    max_retries=3
                )
                self.logger.info(f"✅ Groq LLM initialized: {Config.GROQ_MODEL}")

            elif Config.LLM_TYPE == "ollama":
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=Config.OLLAMA_MODEL,
                    temperature=Config.TEMPERATURE,
                    base_url=Config.OLLAMA_BASE_URL,
                    request_timeout=120
                )
                self.logger.info(f"✅ Ollama LLM initialized: {Config.OLLAMA_MODEL}")

            elif Config.LLM_TYPE == "openai":
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=Config.LLM_MODEL,
                    temperature=Config.TEMPERATURE,
                    openai_api_key=Config.OPENAI_API_KEY,
                    max_tokens=Config.MAX_TOKENS
                )
                self.logger.info(f"✅ OpenAI LLM initialized: {Config.LLM_MODEL}")

            else:
                raise ValueError(f"Unsupported LLM type: {Config.LLM_TYPE}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {str(e)}")
            raise Exception(f"Failed to initialize LLM: {str(e)}")

    def _create_system_prompt(self) -> str:
        """
        Create the comprehensive system prompt for the agent.

        Returns:
            Detailed system prompt string
        """
        llm_info = Config.get_llm_info()
        
        return f"""You are an **Advanced AI Research Assistant** powered by {llm_info['provider']} ({llm_info['model']}) with sophisticated reasoning capabilities and access to multiple information sources. Your core mission is to provide thorough, well-researched, and accurate responses by intelligently combining information from uploaded documents and real-time web search.

## 🧠 **Core Capabilities & Decision Framework**

### **Tool Selection Intelligence**

You have access to two primary tools:

1. **`document_search`** - Search uploaded PDF documents
   - **Use for**: Questions about uploaded content, specific document analysis, quotes from papers, data from reports
   - **Best when**: User asks about "the document", "this paper", "uploaded file", or refers to specific content they've shared

2. **`web_search`** - Search current web information  
   - **Use for**: Current events, latest news, general knowledge, recent developments, verification of facts
   - **Best when**: User asks about "latest", "current", "recent", "what's new", or topics not in documents

### **Decision Logic**

- **Document First**: If user mentions documents or asks about uploaded content → use `document_search`
- **Web for Current**: If user asks about current/recent information → use `web_search`  
- **Combined Intelligence**: For complex queries, use both tools to provide comprehensive answers
- **Context Aware**: Consider conversation history and previous tool usage

## 🎯 **Response Quality Standards**

### **Structure & Clarity**
- Start with direct answers to the user's question
- Organize information logically with clear sections
- Use markdown formatting for better readability
- Provide specific examples and details when relevant

### **Source Attribution**
- **Always** cite your sources clearly
- For documents: Mention file name, page number when available
- For web results: Reference the source and include key details
- Distinguish between document-based and web-based information

### **Analytical Depth**
- Go beyond simple information retrieval
- Provide analysis, insights, and connections between ideas
- Explain implications and significance of findings
- Offer multiple perspectives when appropriate

## 💡 **Conversation Management**

### **Memory & Context**
- Remember previous parts of the conversation
- Build upon earlier discussions
- Reference previous questions and answers when relevant
- Maintain context across multiple exchanges

### **Follow-up Intelligence**
- Anticipate likely follow-up questions
- Suggest related topics or deeper exploration
- Offer to search different sources if needed
- Ask clarifying questions when user intent is unclear

## 🔄 **Multi-Step Reasoning Process**

For complex queries, follow this approach:
1. **Analyze** the user's question and identify information needs
2. **Plan** which tools to use and in what order
3. **Execute** searches systematically
4. **Synthesize** information from multiple sources
5. **Present** a comprehensive, well-organized response

## ⚠️ **Important Guidelines**

### **Accuracy & Honesty**
- If information is not found, clearly state this
- Don't make up or hallucinate information
- Acknowledge limitations and uncertainties
- Distinguish between facts and interpretations

### **User Experience**
- Be conversational yet professional
- Show enthusiasm for helping with research
- Adapt your communication style to the user's needs
- Provide actionable insights and next steps

### **Error Handling**
- If a tool fails, try alternative approaches
- Explain any technical limitations clearly
- Offer workarounds or alternative solutions
- Maintain helpfulness even when facing constraints

**System Info**: Running on {llm_info['deployment']} mode with {llm_info['provider']} ({llm_info['model']})

Remember: You're not just answering questions—you're conducting research, analyzing information, and providing valuable insights. Be thorough, be accurate, and be genuinely helpful in advancing the user's understanding."""

    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """
        Create the agent's prompt template with system message and conversation history.

        Returns:
            ChatPromptTemplate for the agent
        """
        system_message = self._create_system_prompt()

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ])

    def _create_agent_node(self):
        """
        Create the main agent node that processes messages and decides on actions.

        Returns:
            Agent node function
        """
        # Get available tools
        tools = self.tool_manager.get_all_tools()

        # Bind tools to the model
        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
            self.logger.info(f"🛠️  LLM bound with {len(tools)} tools: {[tool.name for tool in tools]}")
        else:
            llm_with_tools = self.llm
            self.logger.warning("⚠️  No tools available for agent")

        # Create the agent prompt
        prompt = self._create_agent_prompt()

        # Create the agent chain
        agent_runnable = prompt | llm_with_tools

        def agent_node(state: AgentState):
            """
            Agent node that processes the current state and generates responses or tool calls.

            Args:
                state: Current agent state with messages and context

            Returns:
                Updated state with new messages
            """
            try:
                self.logger.info("🤖 Agent processing new message...")

                # Prepare the input for the agent
                agent_input = {
                    "messages": state["messages"],
                }

                # Add context about document availability
                if state.get("documents_uploaded", False):
                    # Add a system message about document availability
                    context_msg = ("📄 **Context**: Documents are available for search. "
                                   "Use the document_search tool when users ask about uploaded content.")
                    self.logger.debug("📚 Documents available for search")
                else:
                    context_msg = ("🌐 **Context**: No documents uploaded. "
                                   "Use web_search for all information needs.")
                    self.logger.debug("🌍 No documents available, will use web search only")

                # Get the response from the agent
                response = agent_runnable.invoke(agent_input)
                self.logger.info(f"✅ Agent generated response with {len(getattr(response, 'tool_calls', []))} tool calls")

                return {"messages": [response]}

            except Exception as e:
                self.logger.error(f"❌ Error in agent node: {str(e)}")
                # Create an error response
                error_message = AIMessage(
                    content=(
                        f"I apologize, but I encountered an error while processing your request: {str(e)}. "
                        f"This might be due to API connectivity issues or configuration problems. "
                        f"Please try again, and if the problem persists, check your API key configuration."
                    )
                )
                return {"messages": [error_message]}

        return agent_node

    def _should_continue(self, state: AgentState) -> str:
        """
        Determine whether the agent should continue to use tools or end the conversation turn.

        Args:
            state: Current agent state

        Returns:
            Next step: "tools" or END
        """
        last_message = state["messages"][-1]

        # Check if the last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            self.logger.info(f"🛠️  Agent will use {len(last_message.tool_calls)} tools")
            return "tools"

        # Otherwise, end the conversation turn
        self.logger.info("✅ Agent completed processing, ending turn")
        return END

    def _build_graph(self):
        """
        Build the LangGraph workflow that orchestrates the agent's reasoning process.

        This creates a state machine that handles the flow between agent reasoning
        and tool execution, with proper error handling and state management.
        """
        try:
            self.logger.info("🔧 Building agent graph...")

            # Create the state graph
            workflow = StateGraph(AgentState)

            # Add the main agent node
            agent_node = self._create_agent_node()
            workflow.add_node("agent", agent_node)

            # Get available tools and create tool node if tools exist
            tools = self.tool_manager.get_all_tools()

            if tools:
                # Create tool node with available tools
                tool_node = ToolNode(tools)
                workflow.add_node("tools", tool_node)

                # Add conditional edges from agent
                workflow.add_conditional_edges(
                    "agent",
                    self._should_continue,
                    {
                        "tools": "tools",
                        END: END
                    }
                )

                # Add edge from tools back to agent
                workflow.add_edge("tools", "agent")
                self.logger.info(f"📊 Graph built with {len(tools)} tools")
            else:
                # No tools available, just end after agent
                workflow.add_edge("agent", END)
                self.logger.warning("⚠️  Graph built without tools")

            # Set the entry point
            workflow.set_entry_point("agent")

            # Compile the graph with memory checkpointing
            self.graph = workflow.compile(checkpointer=self.memory)

            self.logger.info("✅ Agent graph built and compiled successfully")

        except Exception as e:
            self.logger.error(f"❌ Error building agent graph: {str(e)}")
            raise Exception(f"Failed to build agent graph: {str(e)}")

    def update_document_processor(self, document_processor: DocumentProcessor):
        """
        Update the document processor and rebuild the graph with new tools.

        Args:
            document_processor: New DocumentProcessor instance
        """
        try:
            self.logger.info("🔄 Updating document processor...")

            # Update the document processor
            self.document_processor = document_processor
            self.tool_manager.update_document_processor(document_processor)

            # Rebuild the graph with updated tools
            self._build_graph()

            self.logger.info("✅ Document processor updated and graph rebuilt successfully")

        except Exception as e:
            self.logger.error(f"❌ Error updating document processor: {str(e)}")
            raise Exception(f"Failed to update document processor: {str(e)}")

    def chat(self, message: str, session_id: str = "default") -> str:
        """
        Process a chat message and return the agent's response.

        This is the main interface for interacting with the agent. It handles
        the complete conversation flow including state management, tool usage,
        and response generation.

        Args:
            message: User's message/question
            session_id: Unique identifier for the conversation session

        Returns:
            Agent's response as a string
        """
        try:
            if not message or not message.strip():
                return "Please provide a message for me to respond to."

            self.logger.info(f"💬 Processing chat message for session {session_id}: '{message[:50]}...'")

            # Update session statistics
            session_data = self.session_manager.get_session(session_id)
            self.session_manager.increment_counter(session_id, "total_queries")
            self.session_manager.increment_counter(session_id, "conversation_turns")

            # Create the input state
            input_state = {
                "messages": [HumanMessage(content=message)],
                "documents_uploaded": self.document_processor is not None,
                "session_id": session_id,
                "user_intent": None,  # Could be enhanced with intent detection
                "tool_results": None
            }

            # Configure for this session
            config = {"configurable": {"thread_id": session_id}}

            # Process through the graph
            result = self.graph.invoke(input_state, config)

            # Extract the response
            if result and "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    response_content = last_message.content

                    # Update session with successful query
                    self.session_manager.increment_counter(session_id, "successful_queries")
                    self.logger.info(f"✅ Generated response for session {session_id} ({len(response_content)} chars)")

                    return response_content
                else:
                    # Handle case where message has no content
                    error_response = ("I generated a response, but it seems to be empty. "
                                      "Please try rephrasing your question or ask something else.")
                    self.session_manager.increment_counter(session_id, "error_count")
                    return error_response
            else:
                # Handle case where no messages in result
                error_response = ("I'm having trouble generating a response right now. "
                                  "Please try again or rephrase your question.")
                self.session_manager.increment_counter(session_id, "error_count")
                return error_response

        except Exception as e:
            self.logger.error(f"❌ Error in chat processing for session {session_id}: {str(e)}")
            # Update error count
            self.session_manager.increment_counter(session_id, "error_count")

            # Return user-friendly error message
            return (f"I apologize, but I encountered an error while processing your message. "
                    f"This could be due to API connectivity issues or system limitations. "
                    f"Please try again, and if the problem persists, check that your API keys "
                    f"are properly configured. Error details: {str(e)}")

    def get_conversation_history(self, session_id: str = "default", limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return (most recent first)

        Returns:
            List of messages in the conversation
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            state = self.graph.get_state(config)

            messages = state.values.get("messages", []) if state.values else []

            if limit and len(messages) > limit:
                messages = messages[-limit:]  # Get most recent messages

            self.logger.debug(f"📜 Retrieved {len(messages)} messages for session {session_id}")
            return messages

        except Exception as e:
            self.logger.error(f"❌ Error getting conversation history for session {session_id}: {str(e)}")
            return []

    def clear_conversation(self, session_id: str = "default") -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a new session ID with timestamp to effectively "clear" the conversation
            new_session_id = f"{session_id}_{datetime.now().timestamp()}"

            # Update session manager
            self.session_manager.create_session(new_session_id)

            self.logger.info(f"🗑️  Conversation cleared for session {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error clearing conversation for session {session_id}: {str(e)}")
            return False

    def get_available_tools(self) -> List[str]:
        """
        Get list of currently available tool names.

        Returns:
            List of tool names that the agent can use
        """
        tools = self.tool_manager.get_all_tools()
        tool_names = [tool.name for tool in tools]
        self.logger.debug(f"🛠️  Available tools: {tool_names}")
        return tool_names

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status information.

        Returns:
            Dictionary with detailed system status
        """
        try:
            # Get LLM info
            llm_info = Config.get_llm_info()

            # Get tool status
            tool_status = self.tool_manager.get_tools_status()

            # Get document processor info
            doc_info = {}
            if self.document_processor:
                doc_info = self.document_processor.get_database_info()

            status = {
                "agent_initialized": self.graph is not None,
                "llm_model": f"{llm_info['provider']}:{llm_info['model']}",
                "llm_type": llm_info['type'],
                "deployment_mode": llm_info['deployment'],
                "embedding_model": Config.HF_EMBEDDING_MODEL,
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_TOKENS,
                "documents_available": self.document_processor is not None,
                "available_tools": self.get_available_tools(),
                "tool_status": tool_status,
                "document_info": doc_info,
                "session_count": len(self.session_manager.sessions),
                "timestamp": datetime.now().isoformat(),
                "config_valid": True  # Assume valid since we got this far
            }

            return status

        except Exception as e:
            self.logger.error(f"❌ Error getting system status: {str(e)}")
            return {
                "agent_initialized": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_session_stats(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Get statistics for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        return self.session_manager.get_session_stats(session_id)

    def test_connectivity(self) -> Dict[str, Any]:
        """
        Test connectivity to external services (LLM, web search, etc.).

        Returns:
            Dictionary with connectivity test results
        """
        results = {
            "llm_connection": {"status": "unknown", "error": None},
            "web_search": {"status": "unknown", "error": None},
            "document_search": {"status": "unknown", "error": None}
        }

        # Test LLM connection
        try:
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            results["llm_connection"]["status"] = "success"
            llm_info = Config.get_llm_info()
            results["llm_connection"]["model"] = f"{llm_info['provider']}:{llm_info['model']}"
            results["llm_connection"]["deployment"] = llm_info['deployment']

        except Exception as e:
            results["llm_connection"]["status"] = "failed"
            results["llm_connection"]["error"] = str(e)

        # Test tools
        try:
            tool_tests = self.tool_manager.test_tools()
            results["web_search"] = tool_tests.get("web_search", results["web_search"])
            results["document_search"] = tool_tests.get("document_search", results["document_search"])

        except Exception as e:
            results["web_search"]["status"] = "failed"
            results["web_search"]["error"] = str(e)

        return results

def test_agent():
    """Test the agent functionality with basic operations."""
    print("🚀 Testing Agentic RAG Assistant...")
    try:
        # Test initialization without document processor
        assistant = AgenticRAGAssistant()
        print("✅ Agent initialized without document processor")

        # Test system status
        status = assistant.get_system_status()
        print(f"✅ System status retrieved: {status.get('agent_initialized', False)}")
        print(f"📊 Available tools: {status.get('available_tools', [])}")
        print(f"🤖 LLM: {status.get('llm_model', 'Unknown')}")
        print(f"🌍 Deployment: {status.get('deployment_mode', 'Unknown')}")

        # Test connectivity (this will make actual API calls if keys are configured)
        api_key_available = False
        if Config.LLM_TYPE == "groq" and Config.GROQ_API_KEY:
            api_key_available = Config.GROQ_API_KEY.startswith("gsk_")
        elif Config.LLM_TYPE == "ollama":
            api_key_available = True  # No API key needed for Ollama
        elif Config.LLM_TYPE == "openai" and Config.OPENAI_API_KEY:
            api_key_available = Config.OPENAI_API_KEY.startswith("sk-")

        if api_key_available:
            print("🔍 Testing connectivity...")
            connectivity = assistant.test_connectivity()
            print(f"📡 LLM Connection: {connectivity['llm_connection']['status']}")
            print(f"🌐 Web Search: {connectivity['web_search']['status']}")
        else:
            print("⚠️  Skipping connectivity test (API keys not configured)")

        # Test a simple chat (only if API keys are properly configured)
        if api_key_available and Config.TAVILY_API_KEY:
            print("💬 Testing chat functionality...")
            response = assistant.chat("Hello! What can you help me with?")
            print(f"🤖 Response: {response[:100]}...")
        else:
            print("⚠️  Skipping chat test (API keys not fully configured)")

        print("🎉 Agent test completed successfully!")
        return assistant

    except Exception as e:
        print(f"❌ Agent test failed: {str(e)}")
        return None

if __name__ == "__main__":
    test_agent()
