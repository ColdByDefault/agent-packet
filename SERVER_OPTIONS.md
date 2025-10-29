# Backend Server Options

Quick comparison of the two backend server options.

## Which Server Should I Use?

### 🎯 Use `api_server.py` (Full Server) if:

- You want MCP tool execution (calculator, datetime, text search, system info)
- You're building a full-featured agent application
- You don't mind using two ports (8001 + 8002)

### 🎯 Use `api_server_simple.py` (Simple Server) if:

- You only need chat functionality
- You want a minimal setup with one port
- You don't need tool execution
- You're just testing or prototyping

---

## Quick Comparison

| Feature             | api_server.py | api_server_simple.py |
| ------------------- | ------------- | -------------------- |
| **Chat with LLM**   | ✅ Yes        | ✅ Yes               |
| **RAG (Knowledge)** | ✅ Yes        | ✅ Yes               |
| **MCP Tools**       | ✅ Yes        | ❌ No                |
| **Health Check**    | ✅ Yes        | ✅ Yes               |
| **Status Endpoint** | ✅ Yes        | ✅ Yes               |
| **API Docs**        | ✅ Yes        | ✅ Yes               |
| **Ports Used**      | 8001 + 8002   | 8001 only            |
| **Restart (rs)**    | ✅ Yes        | ✅ Yes               |
| **Auto-reload**     | ✅ Yes        | ✅ Yes               |

---

## How to Run

### Full Server

```bash
cd llm_local_agent
python api_server.py
```

### Simple Server

```bash
cd llm_local_agent
python api_server_simple.py
```

Both run on **http://localhost:8001** (same port for frontend)

---

## Port Layout

### Full Server (api_server.py)

```
┌──────────────────────────────────┐
│  Frontend (Port 3000)            │
└────────────┬─────────────────────┘
             │
             ↓
┌──────────────────────────────────┐
│  REST API (Port 8001)            │
│  - /health, /chat, /status       │
└────────────┬─────────────────────┘
             │
             ↓
┌──────────────────────────────────┐
│  MCP Internal (Port 8002)        │
│  - Tool execution                │
└──────────────────────────────────┘
```

### Simple Server (api_server_simple.py)

```
┌──────────────────────────────────┐
│  Frontend (Port 3000)            │
└────────────┬─────────────────────┘
             │
             ↓
┌──────────────────────────────────┐
│  REST API (Port 8001)            │
│  - /health, /chat, /status       │
│  - No MCP, no tools              │
└──────────────────────────────────┘
```

---

## Switching Between Servers

1. **Stop** the running server (Ctrl+C)
2. **Run** the other server:
   - `python api_server.py` OR
   - `python api_server_simple.py`
3. **Frontend** automatically connects (both use port 8001)