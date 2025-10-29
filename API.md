# Local LLM Agent API Documentation

Simple API reference for the Local LLM Agent backend.

**Base URL:** `http://localhost:8001` (default MCP server port)

---

## Endpoints

### 1. Server Status

**GET** `/`

Get server status and information.

**Response:**

```json
{
  "name": "Local LLM Agent MCP Server",
  "version": "0.1.0",
  "status": "running"
}
```

---

### 2. Health Check

**GET** `/health`

Check if server is healthy and get tool count.

**Response:**

```json
{
  "status": "healthy",
  "tools": 3
}
```

---

### 3. List Available Tools

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
