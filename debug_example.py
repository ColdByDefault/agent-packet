#!/usr/bin/env python3
"""
Debug version of example.py with more detailed error output.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# Add src to path to import the agent
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_agent import LocalLLMAgent, AgentConfig
from llm_agent.core.config import OllamaConfig, RAGConfig, MCPConfig

async def main():
    """Main example function with detailed error handling."""
    print("🚀 Local LLM Agent Example (Debug Mode)")
    print("=" * 50)
    
    try:
        # Create configuration
        config = AgentConfig(
            agent_name="MyLocalAgent",
            system_prompt="""You are a helpful AI assistant running locally. You have access to:
            - Your local knowledge base for retrieving relevant information
            - Various tools for calculations, text search, and system information
            - The ability to remember our conversation
            
            Always be helpful, accurate, and mention when you're using your knowledge base or tools.""",
            
            # Configure for your local setup
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                model="llama3.1:8b",  # Using your installed model
                temperature=0.7
            ),
            
            rag=RAGConfig(
                vector_db_path="./data/vectordb",
                embedding_model="nomic-embed-text:latest",  # Using your embedding model
                chunk_size=800,
                max_results=5
            ),
            
            mcp=MCPConfig(
                enabled=True,
                server_port=8001
            )
        )
        
        print("✅ Configuration created successfully")
        print(f"   • Ollama URL: {config.ollama.base_url}")
        print(f"   • Model: {config.ollama.model}")
        print(f"   • Embedding: {config.rag.embedding_model}")
        
        # Initialize the agent
        print("\n🔧 Initializing agent components...")
        async with LocalLLMAgent(config) as agent:
            print("✅ Agent initialized successfully!")
            
    except Exception as e:
        print(f"\n❌ Detailed Error Information:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print(f"\n📋 Full Traceback:")
        traceback.print_exc()
        
        print(f"\n🔍 Debugging Information:")
        print(f"   • Python Path: {sys.path[:3]}...")
        print(f"   • Working Directory: {os.getcwd()}")
        print(f"   • Python Version: {sys.version}")

if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Outer Error: {e}")
        traceback.print_exc()