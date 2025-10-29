"""
Simple example demonstrating the Local LLM Agent capabilities.
Run this script to see the agent in action with your local setup.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path to import the agent
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_agent import LocalLLMAgent, AgentConfig
from llm_agent.core.config import OllamaConfig, RAGConfig, MCPConfig


async def main():
    """Main example function."""
    print("🚀 Local LLM Agent Example")
    print("=" * 50)
    
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
    
    # Initialize the agent
    print("🔧 Initializing agent components...")
    async with LocalLLMAgent(config) as agent:
        print("✅ Agent initialized successfully!")
        
        # Show agent stats
        stats = agent.get_stats()
        print(f"\n📊 Agent Stats:")
        print(f"   • Name: {stats['agent_name']}")
        print(f"   • Model: {stats['llm_model']}")
        print(f"   • MCP Enabled: {stats['mcp_enabled']}")
        if stats.get('available_tools'):
            print(f"   • Available Tools: {stats['available_tools']}")
        
        # Add some knowledge to the RAG system
        print("\n📚 Adding knowledge to the system...")
        
        knowledge_texts = [
            """
            Python is a high-level programming language known for its simplicity and readability.
            It was created by Guido van Rossum and first released in 1991. Python supports
            multiple programming paradigms including procedural, object-oriented, and functional programming.
            """,
            """
            Machine Learning is a subset of artificial intelligence that focuses on algorithms
            that can learn from data. Popular ML libraries in Python include scikit-learn,
            TensorFlow, PyTorch, and pandas for data manipulation.
            """,
            """
            Local LLM deployment allows running language models on your own hardware without
            relying on cloud services. Tools like Ollama make it easy to run models like
            Llama, Gemma, and others locally while maintaining privacy and control.
            """
        ]
        
        for i, text in enumerate(knowledge_texts):
            doc_id = await agent.add_knowledge(text, {"topic": f"knowledge_{i+1}"})
            if doc_id:
                print(f"   ✅ Added document: {doc_id[:16]}...")
            else:
                print(f"   ✅ Added document {i+1} (no ID returned)")
        
        # Demonstrate RAG capabilities
        print("\n🔍 Testing knowledge retrieval...")
        results = await agent.search_knowledge("What is Python programming?", k=2)
        print(f"   Found {len(results)} relevant documents")
        for result in results:
            print(f"   • Score: {result['score']:.3f} - {result['content'][:100]}...")
        
        # Test MCP tools
        print("\n🛠️  Testing MCP tools...")
        available_tools = await agent.get_available_tools()
        print(f"   Available tools: {', '.join(available_tools)}")
        
        # Example tool usage
        if "calculator" in available_tools:
            calc_result = await agent.execute_tool("calculator", {"expression": "2 + 3 * 4"})
            print(f"   Calculator: 2 + 3 * 4 = {calc_result}")
        
        if "datetime" in available_tools:
            time_result = await agent.execute_tool("datetime", {"operation": "current"})
            print(f"   Current time: {time_result['formatted']}")
        
        # Interactive chat examples
        print("\n💬 Chat Examples:")
        print("-" * 30)
        
        test_queries = [
            "What is Python and why is it popular?",
            "Tell me about machine learning libraries",
            "Calculate the area of a circle with radius 5 (use π ≈ 3.14159)",
            "What's the current date and time?"
        ]
        
        for query in test_queries:
            print(f"\n👤 User: {query}")
            print("🤖 Assistant: ", end="")
            
            # Use streaming response for better UX
            response_parts = []
            async for chunk in agent.chat_stream(query):
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
            print()  # New line after response
        
        # Show final stats
        final_stats = agent.get_stats()
        print(f"\n📈 Final Stats:")
        print(f"   • Conversation length: {final_stats['conversation_length']} messages")
        
        # MCP server info
        if agent.mcp_server:
            server_info = agent.mcp_server.get_server_info()
            print(f"   • MCP Server: {server_info['url']}")
        
        print("\n🎉 Example completed successfully!")
        print("\n💡 Tips:")
        print("   • The MCP server is running and provides tools via HTTP API")
        print("   • Knowledge is persisted in the vector database")
        print("   • You can extend this by adding custom tools and knowledge")
        print("   • Check the logs for detailed information about operations")


if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure Ollama is running with the required models:")
        print("   • llama3.1:8b")
        print("   • nomic-embed-text:latest")