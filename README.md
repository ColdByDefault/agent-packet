# Local LLM Agent

Local agent system integrating Ollama LLMs, ChromaDB vector storage, and MCP tool protocol.

## Architecture

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

## 1. Clone Repository

```bash
git clone https://github.com/ColdByDefault/agent-packet.git
cd agent-packet
```

## 2. Install Python 3.10

### Manual Download

Download and install Python 3.10 from [python.org](https://www.python.org/downloads/)

### PowerShell (Windows)

```powershell
# Using winget
winget install Python.Python.3.10

# Using chocolatey
choco install python --version=3.10.11

# Using scoop
scoop install python310
```

## 3. Install Miniconda

### Manual Download

Download and install Miniconda from [conda.io](https://docs.conda.io/en/latest/miniconda.html)

### PowerShell (Windows)

```powershell
# Using winget
winget install Anaconda.Miniconda3

# Using chocolatey
choco install miniconda3

# Using scoop
scoop install miniconda3
```

## 4. Install Ollama and Models

### Install Ollama

Download from [ollama.ai](https://ollama.ai/) and install.

### Pull Required Models

```bash
ollama serve
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest
```

### Change Models (Optional)

Edit `.env` file to use different models:

```bash
OLLAMA__MODEL=llama3.2:3b               # Alternative LLM
RAG__EMBEDDING_MODEL=mxbai-embed-large  # Alternative embedding model
```

## 5. Create Conda Environment with Python 3.10

```bash
conda create -n llm_agent python=3.10 -y
conda activate llm_agent
```

## 6. Install Requirements

```bash
pip install -r requirements.txt
```

## 7. Test Setup

### Test Ollama Connection

```bash
python debug/debug_ollama.py
```

Expected output:

- Connection status: 200
- Available models listed
- Chat API test successful

### Test Configuration and Imports

```bash
python debug/test_setup.py
```

Expected output:

- Configuration created
- All imports successful
- Ollama accessible
- Available models listed

### Debug Example (Detailed Errors)

```bash
python debug/debug_example.py
```

## 8. Run Final Test

```bash
python example.py
```

## 9. What Happens When Running example.py

### Process Flow:

1. **Agent Initialization**: Creates LocalLLMAgent with Ollama, ChromaDB, and MCP server
2. **Component Setup**: Initializes LLM provider, vector database, and tool server
3. **Knowledge Addition**: Adds 3 sample documents to vector database
4. **Knowledge Search**: Tests retrieval of relevant documents
5. **Tool Testing**: Tests calculator and datetime tools
6. **Interactive Chat**: Demonstrates 4 example conversations
7. **Statistics Display**: Shows final conversation length and server info

### Expected Results:

- Agent stats showing model and tools
- Vector database populated with 3 documents
- Knowledge retrieval scores and content previews
- Tool execution results (calculations, current time)
- Streaming chat responses for each test query
- MCP server running confirmation

## 10. MCP Tools Available

### Built-in Tools:

- **calculator**: Execute mathematical expressions
  ```python
  await agent.execute_tool("calculator", {"expression": "2 + 3 * 4"})
  ```
- **text_search**: Regex search in text
  ```python
  await agent.execute_tool("text_search", {"text": "sample", "pattern": "sam.*"})
  ```
- **datetime**: Date/time operations
  ```python
  await agent.execute_tool("datetime", {"operation": "current"})
  ```
- **system_info**: System information
  ```python
  await agent.execute_tool("system_info", {"info_type": "platform"})
  ```

### Tool Server:

- **Default Port**: 8001 (in `.env.example`)
- **HTTP API**: `http://localhost:8001`
- **Access**: RESTful endpoints for each tool

## 11. Running the API Server

You have **two options** for running the backend API server:

### Option 1: Full Server with MCP Tools (Recommended)

```bash
python api_server.py
```

**Features:**

- ✅ Chat with LLM
- ✅ RAG (Knowledge retrieval)
- ✅ MCP Tools (calculator, datetime, text_search, system_info)
- ✅ Health check & status endpoints

**Ports Used:**

- `8001` - REST API (frontend connects here)
- `8002` - MCP internal server (for tools)

---

### Option 2: Simple Server without MCP Tools

```bash
python api_server_simple.py
```

**Features:**

- ✅ Chat with LLM
- ✅ RAG (Knowledge retrieval)
- ❌ No MCP Tools
- ✅ Health check & status endpoints

**Ports Used:**

- `8001` - REST API only (no extra ports)

**When to use:** If you don't need tool execution and want a simpler setup with just one port.

---

## 12. Server Configuration

### Port Configuration

The system uses multiple ports for different services:

- **FastAPI REST API Server**: `8001` (Main API for frontend/external connections)
  - Health check: `http://localhost:8001/health`
  - API docs: `http://localhost:8001/docs`
  - Chat endpoint: `http://localhost:8001/chat`
- **MCP Internal Server**: `8002` (Used internally by the agent for tool execution)
  - Configured in `api_server.py`: `server_port=8002`
  - Not directly accessible from frontend
- **Ollama LLM Server**: `11434` (Default Ollama port)
  - Base URL: `http://localhost:11434`

### MCP Server Port

- **Example.py uses**: `8001` (configurable)
- **API Server uses**: `8002` (to avoid conflict with main API on 8001)

```bash
MCP__SERVER_PORT=8001  # Default in .env (for standalone MCP server)
```

**Note**: When running the FastAPI server (`api_server.py`), the MCP server runs on port 8002 internally, while the main REST API uses port 8001.

### Running Without MCP

You can disable MCP and run only the REST API server by setting `enabled=False` in the `MCPConfig`:

```bash
python api_server_simple.py
```

When MCP is disabled:

- Only port **8001** is used (REST API)
- Port **8002** is not opened
- The agent will work without tool execution capabilities
- Chat, health check, and status endpoints still function normally

### Ollama Server Port

Default: `11434`

```bash
OLLAMA__BASE_URL=http://localhost:11434
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Ollama settings
OLLAMA__BASE_URL=http://localhost:11434
OLLAMA__MODEL=llama3.1:8b
OLLAMA__TEMPERATURE=0.7

# RAG settings
RAG__VECTOR_DB_PATH=./data/vectordb
RAG__EMBEDDING_MODEL=nomic-embed-text:latest
RAG__CHUNK_SIZE=1000
RAG__MAX_RESULTS=5

# MCP settings
MCP__ENABLED=true
MCP__SERVER_PORT=8001
```
