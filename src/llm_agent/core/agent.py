"""
Main Local LLM Agent implementation.
Orchestrates LLM, RAG, and MCP components for a complete agent system.
"""

import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from loguru import logger

from .config import AgentConfig
from .persistence import ConversationPersistence
from .memory_helper import MemoryHelper
from ..llm import LLMProvider, LLMMessage, LLMResponse, LLMRole, create_llm_provider
from ..rag import LocalRAGSystem, create_rag_system
from ..mcp import MCPServer, create_basic_mcp_server, ToolCall


class ConversationManager:
    """Manages conversation history and context with persistence and long-term memory."""
    
    def __init__(self, max_length: int = 20, enable_persistence: bool = True):
        """Initialize conversation manager."""
        self.max_length = max_length
        self.messages: List[LLMMessage] = []
        self.system_prompt: Optional[str] = None
        self.enable_persistence = enable_persistence
        self.persistence = ConversationPersistence() if enable_persistence else None
        self.memory: Dict[str, Any] = {}  # Long-term memory
        
        # Load memory if persistence is enabled
        if self.enable_persistence and self.persistence:
            self.memory = self.persistence.load_memory()
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt."""
        self.system_prompt = prompt
    
    def update_memory(self, key: str, value: Any) -> None:
        """Update long-term memory."""
        self.memory[key] = value
        
        # Save memory to disk
        if self.enable_persistence and self.persistence:
            try:
                self.persistence.save_memory(self.memory)
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Get value from long-term memory."""
        return self.memory.get(key, default)
    
    def get_memory_summary(self) -> str:
        """Get a summary of stored memory for context."""
        if not self.memory:
            return ""
        
        summary_parts = []
        for key, value in self.memory.items():
            if isinstance(value, (str, int, float, bool)):
                summary_parts.append(f"- {key}: {value}")
            else:
                summary_parts.append(f"- {key}: [stored]")
        
        if summary_parts:
            return "## Remembered Information:\n" + "\n".join(summary_parts)
        return ""
    
    def add_message(self, role: LLMRole, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add message to conversation."""
        message = LLMMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        
        # Trim conversation if too long
        if len(self.messages) > self.max_length:
            # Keep system message if it exists, then trim from oldest
            if self.messages[0].role == LLMRole.SYSTEM:
                self.messages = [self.messages[0]] + self.messages[-(self.max_length-1):]
            else:
                self.messages = self.messages[-self.max_length:]
        
        # Save to disk after adding message
        if self.enable_persistence:
            self._save_to_disk()
    
    def _save_to_disk(self) -> None:
        """Save conversation to disk."""
        if not self.persistence:
            return
        
        try:
            messages_data = [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "metadata": msg.metadata or {}
                }
                for msg in self.messages
            ]
            self.persistence.save_conversation(messages_data)
        except Exception as e:
            logger.warning(f"Failed to save conversation: {e}")
    
    def load_from_disk(self) -> None:
        """Load conversation from disk."""
        if not self.persistence:
            return
        
        try:
            messages_data = self.persistence.load_conversation()
            self.messages = []
            
            for msg_data in messages_data:
                role = LLMRole(msg_data["role"])
                content = msg_data["content"]
                metadata = msg_data.get("metadata")
                self.messages.append(LLMMessage(role=role, content=content, metadata=metadata))
            
            if self.messages:
                logger.info(f"Loaded {len(self.messages)} messages from disk")
        except Exception as e:
            logger.warning(f"Failed to load conversation: {e}")
    
    def start_new_session(self) -> str:
        """Start a new conversation session. Returns session_id."""
        # Save current conversation before starting new one
        if self.messages and self.enable_persistence and self.persistence:
            self._save_to_disk()
        
        # Clear current messages but keep memory
        self.messages = []
        
        # Start new session in persistence
        if self.enable_persistence and self.persistence:
            session_id = self.persistence.start_new_session()
            logger.info(f"Started new conversation session: {session_id}")
            return session_id
        
        return "no-persistence"
    
    def get_messages(self, include_system: bool = True) -> List[LLMMessage]:
        """Get conversation messages with memory context."""
        messages = []
        
        # Add system prompt with memory context if available
        if include_system and self.system_prompt:
            system_content = self.system_prompt
            
            # Append memory summary to system prompt
            memory_summary = self.get_memory_summary()
            if memory_summary:
                system_content += "\n\n" + memory_summary
            
            messages.append(LLMMessage(role=LLMRole.SYSTEM, content=system_content))
        
        # Add conversation messages, but skip existing system messages
        for msg in self.messages:
            if msg.role != LLMRole.SYSTEM:
                messages.append(msg)
        
        return messages
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        
        # Clear from disk too
        if self.enable_persistence and self.persistence:
            try:
                self.persistence.clear_conversation()
            except Exception as e:
                logger.warning(f"Failed to clear persisted conversation: {e}")
    
    def get_context_length(self) -> int:
        """Estimate token count of conversation."""
        total_chars = sum(len(msg.content) for msg in self.get_messages())
        return total_chars // 4  # Rough estimation


class LocalLLMAgent:
    """Main Local LLM Agent class."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the Local LLM Agent."""
        self.config = config
        self.conversation = ConversationManager(config.max_conversation_length)
        self.conversation.set_system_prompt(config.system_prompt)
        
        # Components (initialized later)
        self.llm_provider: Optional[LLMProvider] = None
        self.rag_system: Optional[LocalRAGSystem] = None
        self.mcp_server: Optional[MCPServer] = None
        
        self._initialized = False
        logger.info(f"Created {config.agent_name} agent")
    
    async def initialize(self) -> None:
        """Initialize all agent components."""
        if self._initialized:
            logger.warning("Agent already initialized")
            return
        
        try:
            # Validate and create paths
            self.config.validate_paths()
            
            # Load conversation history from disk
            logger.info("Loading conversation history...")
            self.conversation.load_from_disk()
            
            # Initialize LLM provider
            logger.info("Initializing LLM provider...")
            self.llm_provider = create_llm_provider("ollama", self.config.ollama.model_dump())
            await self.llm_provider.initialize()
            
            # Initialize RAG system
            logger.info("Initializing RAG system...")
            rag_config = {
                "embedding": {
                    "type": "ollama",
                    "model": self.config.rag.embedding_model,
                    "base_url": self.config.ollama.base_url
                },
                "vector_db": {
                    "type": "chroma",
                    "db_path": self.config.rag.vector_db_path,
                    "collection_name": "agent_documents"
                },
                "processor": {
                    "type": "text",
                    "chunk_size": self.config.rag.chunk_size,
                    "chunk_overlap": self.config.rag.chunk_overlap
                },
                "max_context_tokens": 4000,
                "similarity_threshold": self.config.rag.similarity_threshold
            }
            self.rag_system = create_rag_system(rag_config)
            await self.rag_system.initialize()
            
            # Initialize MCP server if enabled
            if self.config.mcp.enabled:
                logger.info("Initializing MCP server...")
                self.mcp_server = create_basic_mcp_server(
                    port=self.config.mcp.server_port,
                    enable_builtin_tools=True
                )
                await self.mcp_server.start()
            
            self._initialized = True
            logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            raise
    
    async def chat(self, message: str, use_rag: bool = True) -> str:
        """Chat with the agent."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Add user message to conversation
            self.conversation.add_message(LLMRole.USER, message)
            
            # Get RAG context if enabled
            rag_context = ""
            if use_rag and self.rag_system:
                logger.debug("Retrieving RAG context...")
                augmented = await self.rag_system.get_augmented_context(message)
                if augmented["found_results"]:
                    rag_context = f"\n\nRelevant context:\n{augmented['context']}"
                    logger.debug(f"Retrieved {augmented['used_results']} context sources")
            
            # Prepare messages for LLM
            messages = self.conversation.get_messages()
            
            # Enhance last user message with RAG context if available
            if rag_context and messages:
                last_msg = messages[-1]
                if last_msg.role == LLMRole.USER:
                    last_msg.content += rag_context
            
            # Generate response
            logger.debug("Generating LLM response...")
            response = await self.llm_provider.generate(messages)
            
            # Add assistant response to conversation
            self.conversation.add_message(LLMRole.ASSISTANT, response.content)
            
            # Auto-extract and store important information
            extracted_facts = MemoryHelper.auto_extract_facts(message, response.content)
            for key, value in extracted_facts.items():
                self.conversation.update_memory(key, value)
                logger.info(f"Remembered: {key} = {value}")
            
            logger.debug(f"Generated response: {len(response.content)} characters")
            return response.content
            
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            raise
    
    async def chat_stream(self, message: str, use_rag: bool = True) -> AsyncGenerator[str, None]:
        """Stream chat response from the agent."""
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Add user message to conversation
            self.conversation.add_message(LLMRole.USER, message)
            
            # Get RAG context if enabled
            rag_context = ""
            if use_rag and self.rag_system:
                logger.debug("Retrieving RAG context...")
                augmented = await self.rag_system.get_augmented_context(message)
                if augmented["found_results"]:
                    rag_context = f"\n\nRelevant context:\n{augmented['context']}"
            
            # Prepare messages for LLM
            messages = self.conversation.get_messages()
            
            # Enhance last user message with RAG context if available
            if rag_context and messages:
                last_msg = messages[-1]
                if last_msg.role == LLMRole.USER:
                    last_msg.content += rag_context
            
            # Generate streaming response
            logger.debug("Generating streaming LLM response...")
            full_response = ""
            async for chunk in self.llm_provider.generate_stream(messages):
                full_response += chunk
                yield chunk
            
            # Add complete response to conversation
            self.conversation.add_message(LLMRole.ASSISTANT, full_response)
            
        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            raise
    
    async def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add knowledge to the RAG system."""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        try:
            doc_id = await self.rag_system.add_text(text, metadata)
            logger.info(f"Added knowledge document: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add knowledge: {str(e)}")
            raise
    
    async def add_knowledge_from_file(self, file_path: str) -> str:
        """Add knowledge from file to the RAG system."""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        try:
            doc_id = await self.rag_system.add_document_from_file(file_path)
            logger.info(f"Added knowledge from file: {file_path}")
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add knowledge from file: {str(e)}")
            raise
    
    async def search_knowledge(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base."""
        if not self.rag_system:
            raise RuntimeError("RAG system not initialized")
        
        try:
            results = await self.rag_system.search_with_threshold(query, k)
            return [
                {
                    "content": result.chunk.content,
                    "score": result.score,
                    "metadata": result.chunk.metadata
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Knowledge search failed: {str(e)}")
            raise
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute an MCP tool."""
        if not self.mcp_server:
            raise RuntimeError("MCP server not initialized or enabled")
        
        try:
            tool_call = ToolCall(tool_name=tool_name, parameters=parameters)
            result = await self.mcp_server.tool_registry.execute_tool(tool_call)
            
            if result.success:
                logger.info(f"Tool {tool_name} executed successfully")
                return result.result
            else:
                logger.error(f"Tool {tool_name} failed: {result.error}")
                raise RuntimeError(result.error)
                
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            raise
    
    async def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools."""
        if not self.mcp_server:
            return []
        
        return self.mcp_server.tool_registry.get_available_tools()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "agent_name": self.config.agent_name,
            "initialized": self._initialized,
            "conversation_length": len(self.conversation.messages),
            "llm_model": self.config.ollama.model if self.llm_provider else None,
            "mcp_enabled": self.config.mcp.enabled,
        }
        
        if self.mcp_server:
            stats["available_tools"] = len(self.mcp_server.tool_registry.get_available_tools())
        
        return stats
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation.clear()
        logger.info("Conversation history cleared")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        logger.info("Cleaning up agent resources...")
        
        try:
            if self.mcp_server:
                await self.mcp_server.stop()
            
            if self.llm_provider:
                await self.llm_provider.cleanup()
            
            logger.info("Agent cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()