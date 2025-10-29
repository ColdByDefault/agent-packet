# Local LLM Agent API Documentation

Simple API reference for the Local LLM Agent backend.

## Port Configuration

The system uses multiple ports:

- **REST API Server** (Primary): `http://localhost:8001`
  - Main API for frontend/external connections
  - Endpoints: `/health`, `/chat`, `/status`, etc.
  - Run with: `python api_server.py`
- **MCP Internal Server**: `http://localhost:8002`

  - Used internally by the agent for tool execution
  - Not directly accessible from frontend
  - Automatically configured in `api_server.py`

- **Ollama LLM Server**: `http://localhost:11434`
  - Default Ollama installation port

**Base URL for Frontend:** `http://localhost:8001`

---

## REST API Endpoints

### 1. Root

**GET** `/`

Get API server information.

**Response:**

```json
{
  "message": "Local LLM Agent API",
  "status": "running",
  "version": "1.0.0"
}
```

---

### 2. Health Check

**GET** `/health`

**GET** `/health`

Check if the API server is running and get agent status.

**Response:**

```json
{
  "status": "healthy",
  "agent_initialized": true,
  "agent_name": "LocalAgent",
  "timestamp": 1730203456.789
}
```

---

### 3. Agent Status

**GET** `/status`

Get detailed agent status and configuration.

**Response:**

```json
{
  "status": "ready",
  "agent_name": "LocalAgent",
  "llm_model": "llama3.1:8b",
  "ollama_url": "http://localhost:11434",
  "rag_enabled": true,
  "mcp_enabled": true,
  "conversation_length": 5
}
```

---

### 4. Chat with Agent

**POST** `/chat`

Send a message to the LLM agent and get a response.

**Request Body:**

```json
{
  "message": "What is the weather like?"
}
```

**Response:**

```json
{
  "response": "I don't have access to real-time weather information...",
  "metadata": {
    "model": "llama3.1:8b",
    "tokens_used": 150
  }
}
```

---

## MCP Internal Endpoints (Port 8002)

The following endpoints are used internally by the agent and are not directly accessible from the frontend:

### List Available Tools

**GET** `/tools`

Get all available MCP tools.

**Response:**

```json
{
  "tools": [
    {
      "name": "calculator",
      "description": "Perform calculations",
      "parameters": {...}
    },
    {
      "name": "search",
      "description": "Search knowledge base",
      "parameters": {...}
    }
  ]
}
```

---

### 4. Execute Tool

**POST** `/tools/execute`

Execute a specific tool with parameters.

**Request Body:**

```json
{
  "tool_name": "calculator",
  "parameters": {
    "expression": "2 + 2"
  },
  "call_id": "optional-unique-id"
}
```

**Response:**

```json
{
  "success": true,
  "result": 4,
  "error": null,
  "call_id": "optional-unique-id"
}
```

**Error Response:**

```json
{
  "success": false,
  "result": null,
  "error": "Error message here",
  "call_id": "optional-unique-id"
}
```

---

## Agent Features (via Python SDK)

These features are available when using the agent directly in Python:

### Chat

```python
response = await agent.chat("Your message here")
```

### Streaming Chat

```python
async for chunk in agent.chat_stream("Your message"):
    print(chunk, end="")
```

### Add Knowledge

```python
doc_id = await agent.add_knowledge("Your text content", metadata={"source": "doc1"})
```

### Search Knowledge

```python
results = await agent.search_knowledge("query", k=5)
```

---

## Configuration

Default configuration values:

| Setting         | Default                 | Description         |
| --------------- | ----------------------- | ------------------- |
| MCP Server Port | 8001                    | HTTP API port       |
| Ollama URL      | http://localhost:11434  | LLM provider        |
| Default Model   | llama3.1:8b             | Ollama model        |
| Vector DB Path  | ./data/vectordb         | ChromaDB storage    |
| Embedding Model | nomic-embed-text:latest | For RAG             |
| CORS            | Enabled (\*)            | All origins allowed |

---

## Error Codes

| Status | Description                          |
| ------ | ------------------------------------ |
| 200    | Success                              |
| 400    | Bad Request (missing parameters)     |
| 500    | Server Error (tool execution failed) |

---

## Example Usage (JavaScript/TypeScript)

```typescript
// Health check
const health = await fetch("http://localhost:8001/health");
const data = await health.json();

// Execute tool
const response = await fetch("http://localhost:8001/tools/execute", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    tool_name: "calculator",
    parameters: { expression: "10 * 5" },
  }),
});
const result = await response.json();
```

---

## Notes

- All endpoints support CORS for frontend integration
- Server must be started before API calls
- Tools are extensible via Python MCP protocol
- Streaming chat requires WebSocket or SSE (future implementation)
