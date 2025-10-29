"""
Quick test to verify the agent system works.
Run this before the full example to check your setup.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_agent.core.config import AgentConfig


async def test_basic_setup():
    """Test basic configuration and imports."""
    print("🔧 Testing Local LLM Agent Setup")
    print("=" * 40)
    
    try:
        # Test configuration
        print("1. Testing configuration...")
        config = AgentConfig()
        print(f"   ✅ Default config created: {config.agent_name}")
        
        # Test imports
        print("2. Testing imports...")
        from llm_agent import LocalLLMAgent
        print("   ✅ Main imports successful")
        
        from llm_agent.llm import create_llm_provider
        from llm_agent.rag import create_rag_system
        from llm_agent.mcp import create_basic_mcp_server
        print("   ✅ Factory imports successful")
        
        # Test Ollama connection
        print("3. Testing Ollama connection...")
        llm_provider = create_llm_provider("ollama", config.ollama.model_dump())
        
        try:
            await llm_provider.initialize()
            health = await llm_provider.health_check()
            if health:
                print("   ✅ Ollama is accessible")
                models = await llm_provider.get_available_models()
                print(f"   📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                print("   ⚠️  Ollama health check failed")
        except Exception as e:
            print(f"   ❌ Ollama connection failed: {e}")
            print("      Make sure Ollama is running: ollama serve")
        finally:
            await llm_provider.cleanup()
        
        print("\n🎉 Basic setup test completed!")
        print("\n💡 Next steps:")
        print("   • Run 'python example.py' for full demonstration")
        print("   • Make sure you have required models:")
        print("     - ollama pull llama3.1:8b")
        print("     - ollama pull nomic-embed-text:latest")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct conda environment:")
        print("conda activate llm_agent")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(test_basic_setup())