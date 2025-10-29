#!/usr/bin/env python3
"""
Debug script to test Ollama connection and model availability.
"""

import asyncio
import httpx

async def test_ollama():
    """Test Ollama connection and model availability."""
    base_url = "http://localhost:11434"
    
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
            # Test health
            print("🔍 Testing Ollama connection...")
            response = await client.get("/api/tags")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                models_data = response.json()
                models = [model["name"] for model in models_data.get("models", [])]
                print(f"   Available models: {models}")
                
                # Check if our models exist
                required_models = ["llama3.1:8b", "nomic-embed-text:latest"]
                for model in required_models:
                    if model in models:
                        print(f"   ✅ {model} - Found")
                    else:
                        print(f"   ❌ {model} - NOT FOUND")
                        print(f"      Available similar: {[m for m in models if model.split(':')[0] in m]}")
                
                # Test chat with llama3.1:8b
                print("\n🧪 Testing chat API...")
                chat_payload = {
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
                
                chat_response = await client.post("/api/chat", json=chat_payload)
                print(f"   Chat API Status: {chat_response.status_code}")
                
                if chat_response.status_code == 200:
                    result = chat_response.json()
                    content = result.get("message", {}).get("content", "")
                    print(f"   Response: {content[:50]}...")
                else:
                    print(f"   Error: {chat_response.text}")
            else:
                print(f"   Connection failed: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())