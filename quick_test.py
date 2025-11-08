#!/usr/bin/env python3
"""
Quick test to verify the backend is running
"""

import requests
import time

def test_backend():
    print("🔍 Checking if backend is running...")
    
    try:
        # Test if server is accessible
        response = requests.get("http://localhost:8000/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and accessible!")
            
            # Test a simple chat endpoint
            print("🧪 Testing basic chat functionality...")
            payload = {
                "question": "Hello, are you working?",
                "session_id": "test_123"
            }
            
            chat_response = requests.post("http://localhost:8000/chat", json=payload, timeout=15)
            if chat_response.status_code == 200:
                data = chat_response.json()
                print("✅ Chat endpoint working!")
                print(f"📝 Response: {data.get('response', 'No response')[:100]}...")
                return True
            else:
                print(f"❌ Chat endpoint failed: {chat_response.status_code}")
                return False
                
        else:
            print(f"❌ Backend not accessible. Status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure it's running on http://localhost:8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ Backend is taking too long to respond")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if test_backend():
        print("\n🎉 Backend is working! You can now run the full test suite with: python test_api.py")
    else:
        print("\n⚠️  Backend issues detected. Please check if uvicorn is running.")