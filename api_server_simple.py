"""
Simple FastAPI server for the Local LLM Agent (without MCP tools).
Provides REST API endpoints for chat functionality only.
"""

import asyncio
import sys
import threading
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path to import the agent
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_agent import LocalLLMAgent, AgentConfig
from llm_agent.core.config import OllamaConfig, RAGConfig, MCPConfig

# Global agent instance
agent: LocalLLMAgent | None = None
restart_flag = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global agent
    
    # Startup - Silent initialization
    config = AgentConfig(
        agent_name="SimpleAgent",
        system_prompt="""You are a helpful AI assistant running locally. 
        You can chat and answer questions using your training data.
        You have access to your local knowledge base for retrieving relevant information.
        You remember our conversation history even after restarts.
        
        Always be helpful, accurate, and conversational.""",
        
        ollama=OllamaConfig(
            base_url="http://localhost:11434",
            model="llama3.1:8b",
            temperature=0.7
        ),
        
        rag=RAGConfig(
            vector_db_path="./data/vectordb",
            embedding_model="nomic-embed-text:latest",
            chunk_size=800,
            max_results=5
        ),
        
        mcp=MCPConfig(
            enabled=False,  # MCP disabled - no tools, no extra port
            server_port=8002
        )
    )
    
    # Create and initialize agent
    agent = LocalLLMAgent(config)
    
    try:
        await agent.initialize()
    except Exception as e:
        print(f"⚠️  Warning: Could not fully initialize agent: {e}")
    
    yield
    
    # Shutdown
    if agent:
        await agent.cleanup()


# Create FastAPI app with lifespan
app = FastAPI(
    title="Simple LLM Agent API (No MCP)",
    description="REST API for chatting with Local LLM - No tool execution",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

restart_flag = False


def listen_for_restart():
    """Listen for 'rs' command to restart the server."""
    global restart_flag
    
    while not restart_flag:
        try:
            user_input = input().strip().lower()
            if user_input == "rs":
                restart_flag = True
                import os
                os._exit(0)
        except (EOFError, KeyboardInterrupt):
            break


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    metadata: Dict[str, Any] = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple LLM Agent API (No MCP)",
        "status": "running",
        "version": "1.0.0",
        "mcp_enabled": False
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend is running."""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None and agent._initialized,
        "agent_name": agent.config.agent_name if agent else None,
        "mcp_enabled": False,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the agent.
    
    Args:
        request: Chat request with user message
        
    Returns:
        Agent's response
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    try:
        # Chat with the agent
        response_text = await agent.chat(request.message)
        
        return ChatResponse(
            response=response_text,
            metadata={
                "model": agent.config.ollama.model,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/status")
async def get_status():
    """Get detailed agent status."""
    if not agent:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready" if agent._initialized else "initializing",
        "agent_name": agent.config.agent_name,
        "llm_model": agent.config.ollama.model,
        "ollama_url": agent.config.ollama.base_url,
        "rag_enabled": agent.rag_system is not None,
        "mcp_enabled": False,
        "mcp_tools": None,
        "conversation_length": len(agent.conversation.messages)
    }


@app.get("/conversation")
async def get_conversation():
    """Get the current conversation history."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    # Get messages and convert to serializable format
    messages = []
    for msg in agent.conversation.messages:
        messages.append({
            "role": msg.role.value,  # Convert enum to string
            "content": msg.content,
            "timestamp": None,  # Backend doesn't store timestamps
        })
    
    return {
        "messages": messages,
        "conversation_length": len(messages)
    }


@app.delete("/conversation")
async def clear_conversation():
    """Clear the conversation history."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    agent.conversation.clear()
    
    return {
        "message": "Conversation cleared",
        "conversation_length": 0
    }


@app.post("/conversation/new")
async def start_new_conversation():
    """Start a new conversation session, preserving agent's memory."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    # Start new session (clears conversation but keeps memory)
    session_id = agent.conversation.start_new_session()
    
    return {
        "message": "New conversation session started. Agent memory preserved.",
        "session_id": session_id,
        "conversation_length": 0,
        "memory_preserved": True,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.get("/knowledge/stats")
async def get_knowledge_stats():
    """Get statistics about the knowledge base (RAG)."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    if not agent.rag_system:
        return {
            "enabled": False,
            "document_count": 0,
            "chunk_count": 0,
            "embedding_model": None
        }
    
    try:
        # Get collection info from ChromaDB
        collection = agent.rag_system.vector_store.collection
        count = collection.count()
        
        return {
            "enabled": True,
            "chunk_count": count,
            "embedding_model": agent.config.rag.embedding_model,
            "vector_db_path": agent.config.rag.vector_db_path,
            "chunk_size": agent.config.rag.chunk_size,
            "similarity_threshold": agent.config.rag.similarity_threshold
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": str(e),
            "chunk_count": 0
        }


@app.get("/knowledge/search")
async def search_knowledge(query: str, limit: int = 10):
    """Search the knowledge base."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    if not agent.rag_system:
        raise HTTPException(status_code=400, detail="RAG system not enabled")
    
    try:
        results = await agent.search_knowledge(query, k=limit)
        return {
            "query": query,
            "results": results,
            "result_count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory statistics."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    return {
        "conversation_length": len(agent.conversation.messages),
        "max_conversation_length": agent.conversation.max_length,
        "estimated_tokens": agent.conversation.get_context_length(),
        "persistence_enabled": agent.conversation.enable_persistence,
        "system_prompt_set": agent.conversation.system_prompt is not None
    }


def main():
    """Run the API server."""
    print("🤖 Simple LLM Agent API - http://localhost:8001")
    print("📖 Docs: http://localhost:8001/docs\n")
    
    # Start input listener in a separate thread
    input_thread = threading.Thread(target=listen_for_restart, daemon=True)
    input_thread.start()
    
    uvicorn.run(
        "api_server_simple:app",
        host="0.0.0.0",
        port=8001,
        log_level="warning",  # Reduce noise
        reload=True,
        reload_dirs=["./src", "."]
    )


if __name__ == "__main__":
    main()
