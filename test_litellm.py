#!/usr/bin/env python3

import requests
import json

def test_litellm_direct():
    """Test LiteLLM by making a direct API call"""
    
    # Test data
    url = "http://localhost:4000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer bedrock-graphgen-2024",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3-2-3b",
        "messages": [
            {"role": "user", "content": "Hello! This is a test."}
        ],
        "max_tokens": 50
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - LiteLLM server not running")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_endpoint():
    """Test LiteLLM health endpoint"""
    try:
        headers = {"Authorization": "Bearer bedrock-graphgen-2024"}
        response = requests.get("http://localhost:4000/health", headers=headers, timeout=10)
        print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Healthy endpoints: {health_data.get('healthy_count', 0)}")
            print(f"❌ Unhealthy endpoints: {health_data.get('unhealthy_count', 0)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Health check failed - server not running")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing LiteLLM Integration...")
    
    print("\n1. Testing health endpoint...")
    health_ok = test_health_endpoint()
    
    if health_ok:
        print("\n2. Testing chat completions...")
        chat_ok = test_litellm_direct()
        
        if chat_ok:
            print("\n✅ LiteLLM integration working!")
        else:
            print("\n❌ Chat completions failed")
    else:
        print("\n❌ LiteLLM server not accessible")
        print("\nTo start LiteLLM server:")
        print("cd /Users/ldodda/Documents/Codes/GraphGen")
        print("litellm --config litellm_config.yaml &")
