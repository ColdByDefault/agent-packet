# Local LLM Agent

A powerful, scalable local LLM agent system that combines:

- **Local LLM integration** via Ollama
- **RAG (Retrieval Augmented Generation)** with vector databases
- **MCP (Model Context Protocol)** for tool integration
- **Modular architecture** with factory patterns for extensibility

## 🚀 Features

- **Local-First**: Runs entirely on your hardware using Ollama
- **RAG Integration**: ChromaDB vector database for knowledge retrieval
- **Tool Integration**: MCP server with built-in tools (calculator, search, datetime, etc.)
- **Streaming Support**: Real-time streaming responses
- **Extensible**: Plugin architecture for custom tools and providers
- **Type-Safe**: Full type hints with Pydantic configuration
- **Async-First**: Built for performance with async/await

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Provider  │    │   RAG System    │    │   MCP Server    │
│   (Ollama)      │    │   (ChromaDB)    │    │   (Tools)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Local LLM Agent │
                    │  (Orchestrator) │
                    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Required models:
  - `llama3.1:8b` (or your preferred LLM)
  - `nomic-embed-text:latest` (for embeddings)

## 🛠️ Installation

1. **Clone and setup:**

   ```bash
   cd your_project_directory
   conda create -n llm_agent python=3.10 -y
   conda activate llm_agent
   pip install -r requirements.txt
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text:latest
   ```

## 🚀 Quick Start

```python
import asyncio
from src.llm_agent import LocalLLMAgent, AgentConfig

async def main():
    # Create configuration
    config = AgentConfig()

    # Initialize agent
    async with LocalLLMAgent(config) as agent:
        # Add knowledge
        await agent.add_knowledge("Python is a programming language...")

        # Chat with the agent
        response = await agent.chat("What is Python?")
        print(response)

        # Use tools
        result = await agent.execute_tool("calculator", {"expression": "2 + 3"})
        print(f"2 + 3 = {result}")

asyncio.run(main())
```

## 📖 Example Usage

Run the complete example:

```bash
python example.py
```

This will demonstrate:

- Agent initialization
- Knowledge base population
- RAG-powered conversations
- MCP tool usage
- Streaming responses

## 🔧 Configuration

The system uses Pydantic for configuration management. Key settings:

```python
from src.llm_agent.core.config import AgentConfig

config = AgentConfig(
    # Agent settings
    agent_name="MyAgent",
    system_prompt="You are a helpful assistant...",

    # Ollama settings
    ollama=AgentConfig.OllamaConfig(
        model="llama3.1:8b",
        temperature=0.7
    ),

    # RAG settings
    rag=AgentConfig.RAGConfig(
        vector_db_path="./data/vectordb",
        embedding_model="nomic-embed-text:latest"
    ),

    # MCP settings
    mcp=AgentConfig.MCPConfig(
        enabled=True,
        server_port=8000
    )
)
```

## 🧩 Extending the System

### Adding Custom LLM Providers

```python
from src.llm_agent.llm.base import LLMProvider
from src.llm_agent.llm.factory import LLMProviderFactory

class CustomProvider(LLMProvider):
    # Implement abstract methods
    pass

# Register the provider
LLMProviderFactory.register_provider("custom", CustomProvider)
```

### Adding Custom Tools

```python
from src.llm_agent.mcp.base import MCPTool, ToolDefinition

class CustomTool(MCPTool):
    def get_definition(self) -> ToolDefinition:
        # Define tool interface
        pass

    async def execute(self, parameters):
        # Implement tool logic
        pass

# Register with MCP server
agent.mcp_server.register_tool(CustomTool())
```

## 🏷️ Available Tools

Built-in MCP tools:

- **Calculator**: Mathematical expressions
- **Text Search**: Regex search in text
- **DateTime**: Date/time operations
- **System Info**: System information retrieval

## 📁 Project Structure

```
src/llm_agent/
├── core/
│   ├── config.py      # Configuration management
│   └── agent.py       # Main agent orchestrator
├── llm/
│   ├── base.py        # LLM provider abstractions
│   ├── ollama.py      # Ollama integration
│   └── factory.py     # LLM provider factory
├── rag/
│   ├── base.py        # RAG system abstractions
│   ├── chroma.py      # ChromaDB integration
│   ├── embeddings.py  # Ollama embeddings
│   ├── processors.py  # Document processing
│   └── factory.py     # RAG system factory
└── mcp/
    ├── base.py        # MCP abstractions
    ├── tools.py       # Built-in tools
    ├── server.py      # HTTP MCP server
    └── factory.py     # MCP factory
```

## 🔍 API Reference

### LocalLLMAgent

Main agent class that orchestrates all components.

```python
async with LocalLLMAgent(config) as agent:
    # Chat methods
    response = await agent.chat("Hello")
    async for chunk in agent.chat_stream("Hello"):
        print(chunk)

    # Knowledge management
    doc_id = await agent.add_knowledge("Some text...")
    doc_id = await agent.add_knowledge_from_file("document.txt")
    results = await agent.search_knowledge("query")

    # Tool execution
    result = await agent.execute_tool("calculator", {"expression": "2+2"})
    tools = await agent.get_available_tools()

    # Utilities
    stats = agent.get_stats()
    agent.clear_conversation()
```

## 🌟 Why This Architecture?

1. **Modularity**: Each component (LLM, RAG, MCP) is separate and swappable
2. **Extensibility**: Factory patterns make adding new providers easy
3. **Type Safety**: Pydantic ensures configuration validity
4. **Performance**: Async-first design for concurrent operations
5. **Local-First**: No external API dependencies
6. **Production-Ready**: Proper error handling, logging, and resource management

## 🚧 Next Steps

- Add more document processors (PDF, Word, etc.)
- Implement conversation memory persistence
- Add web search tools
- Create CLI interface
- Add Docker deployment
- Implement agent-to-agent communication

## 🤝 Contributing

This is a template project. Customize and extend it for your needs:

1. Add your own tools in `src/llm_agent/mcp/tools.py`
2. Create custom LLM providers in `src/llm_agent/llm/`
3. Implement new vector databases in `src/llm_agent/rag/`
4. Extend configuration in `src/llm_agent/core/config.py`

## 📄 License

MIT License - feel free to use this as a starting point for your projects!
