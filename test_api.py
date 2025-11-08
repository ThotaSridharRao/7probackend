#!/usr/bin/env python3
"""
Comprehensive API Test Suite for DataAnalystBot
Tests all endpoints and core functionality
"""

import requests
import json
import base64
import time
import os
from io import StringIO
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_SESSION_ID = "test_session_123"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")

def print_test(test_name):
    print(f"\n{Colors.YELLOW}🧪 Testing: {test_name}{Colors.END}")

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message):
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.END}")

def create_test_csv():
    """Create a test CSV file for testing"""
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'Salary': [50000, 60000, 70000, 55000, 65000],
        'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing'],
        'Experience': [2, 5, 8, 3, 6]
    }
    df = pd.DataFrame(data)
    csv_string = df.to_csv(index=False)
    return base64.b64encode(csv_string.encode()).decode()

def create_test_image():
    """Create a simple test image (1x1 pixel PNG)"""
    # Minimal PNG data for a 1x1 transparent pixel
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return base64.b64encode(png_data).decode()

def test_server_health():
    """Test if the server is running"""
    print_test("Server Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print_success("Server is running and accessible")
            return True
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Cannot connect to server: {e}")
        return False

def test_chat_endpoint():
    """Test the basic chat endpoint"""
    print_test("Chat Endpoint")
    
    payload = {
        "question": "What is data analysis?",
        "session_id": TEST_SESSION_ID,
        "chat_history": []
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                print_success("Chat endpoint working correctly")
                print_info(f"Response preview: {data['response'][:100]}...")
                return True
            else:
                print_error("Chat endpoint returned empty response")
                return False
        else:
            print_error(f"Chat endpoint failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Chat endpoint request failed: {e}")
        return False

def test_csv_upload_endpoint():
    """Test CSV upload and analysis"""
    print_test("CSV Upload Endpoint")
    
    csv_data = create_test_csv()
    payload = {
        "question": "Analyze this employee data and tell me about salary trends",
        "session_id": TEST_SESSION_ID,
        "chat_history": [],
        "csv_base64": csv_data,
        "csv_filename": "test_employees.csv"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/csv-upload", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                print_success("CSV upload endpoint working correctly")
                print_info(f"Response preview: {data['response'][:100]}...")
                return True
            else:
                print_error("CSV upload endpoint returned empty response")
                return False
        else:
            print_error(f"CSV upload endpoint failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"CSV upload endpoint request failed: {e}")
        return False

def test_image_upload_endpoint():
    """Test image upload and analysis"""
    print_test("Image Upload Endpoint")
    
    image_data = create_test_image()
    payload = {
        "question": "What do you see in this image?",
        "session_id": TEST_SESSION_ID,
        "chat_history": [],
        "image_base64": image_data,
        "image_type": "image/png"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/image-upload", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                print_success("Image upload endpoint working correctly")
                print_info(f"Response preview: {data['response'][:100]}...")
                return True
            else:
                print_error("Image upload endpoint returned empty response")
                return False
        else:
            print_error(f"Image upload endpoint failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Image upload endpoint request failed: {e}")
        return False

def test_multi_upload_endpoint():
    """Test multi-file upload endpoint"""
    print_test("Multi-Upload Endpoint")
    
    csv_data = create_test_csv()
    image_data = create_test_image()
    
    payload = {
        "question": "Analyze both the CSV data and the image I uploaded",
        "session_id": TEST_SESSION_ID,
        "chat_history": [],
        "csv_base64": csv_data,
        "csv_filename": "test_employees.csv",
        "image_base64": image_data,
        "image_type": "image/png"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/multi-upload", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                print_success("Multi-upload endpoint working correctly")
                print_info(f"Response preview: {data['response'][:100]}...")
                return True
            else:
                print_error("Multi-upload endpoint returned empty response")
                return False
        else:
            print_error(f"Multi-upload endpoint failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Multi-upload endpoint request failed: {e}")
        return False

def test_data_analysis_endpoint():
    """Test comprehensive data analysis endpoint"""
    print_test("Data Analysis Endpoint")
    
    csv_data = create_test_csv()
    payload = {
        "csv_base64": csv_data,
        "csv_filename": "test_employees.csv",
        "session_id": TEST_SESSION_ID
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze-data", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["success", "cleaning_log", "statistical_summary", "visualizations"]
            
            if all(field in data for field in required_fields):
                print_success("Data analysis endpoint working correctly")
                print_info(f"Analysis completed successfully: {data.get('success', False)}")
                print_info(f"Cleaning steps: {len(data.get('cleaning_log', []))}")
                print_info(f"Visualizations: {len(data.get('visualizations', {}))}")
                return True
            else:
                print_error("Data analysis endpoint missing required fields")
                return False
        else:
            print_error(f"Data analysis endpoint failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Data analysis endpoint request failed: {e}")
        return False

def test_clean_data_endpoint():
    """Test data cleaning endpoint (fallback)"""
    print_test("Data Cleaning Endpoint")
    
    csv_data = create_test_csv()
    payload = {
        "csv_base64": csv_data,
        "csv_filename": "test_employees.csv",
        "session_id": TEST_SESSION_ID
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/clean-data", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            required_fields = ["success", "cleaning_log", "statistical_summary"]
            
            if all(field in data for field in required_fields):
                print_success("Data cleaning endpoint working correctly")
                print_info(f"Cleaning completed successfully: {data.get('success', False)}")
                return True
            else:
                print_error("Data cleaning endpoint missing required fields")
                return False
        else:
            print_error(f"Data cleaning endpoint failed with status: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Data cleaning endpoint request failed: {e}")
        return False

def test_session_management():
    """Test session management endpoints"""
    print_test("Session Management")
    
    # Test saving chat
    chat_data = {
        "session_id": TEST_SESSION_ID,
        "chat_history": [
            {"type": "human", "content": "Hello"},
            {"type": "ai", "content": "Hi there!"}
        ]
    }
    
    try:
        # Save chat
        response = requests.post(f"{API_BASE_URL}/save-chat", json=chat_data, timeout=10)
        if response.status_code == 200:
            print_success("Chat saving working correctly")
        else:
            print_error(f"Chat saving failed: {response.status_code}")
            return False
        
        # Get recent chats
        response = requests.get(f"{API_BASE_URL}/recent-chat-titles", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "sessions" in data:
                print_success("Recent chat titles retrieval working correctly")
                print_info(f"Found {len(data['sessions'])} chat sessions")
            else:
                print_error("Recent chat titles missing 'sessions' field")
                return False
        else:
            print_error(f"Recent chat titles failed: {response.status_code}")
            return False
        
        # Get specific chat history
        response = requests.get(f"{API_BASE_URL}/recent-chats/{TEST_SESSION_ID}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "chat_history" in data:
                print_success("Chat history retrieval working correctly")
                print_info(f"Retrieved {len(data['chat_history'])} messages")
                return True
            else:
                print_error("Chat history missing 'chat_history' field")
                return False
        else:
            print_error(f"Chat history retrieval failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Session management request failed: {e}")
        return False

def test_core_modules():
    """Test core module imports"""
    print_test("Core Module Imports")
    
    try:
        # Test RAG chain
        from chains.rag_chain import build_chain
        print_success("RAG chain module imported successfully")
        
        # Test data analyzer
        from utils.data_analyzer import DataAnalyzer
        print_success("Data analyzer module imported successfully")
        
        # Test vector database
        from vector_db.faiss_db import EMBEDDING
        print_success("Vector database module imported successfully")
        
        # Test loaders
        from loaders.load_data import load_jsonl
        from loaders.load_csv import load_csv
        print_success("Data loaders imported successfully")
        
        return True
        
    except Exception as e:
        print_error(f"Core module import failed: {e}")
        return False

def run_performance_test():
    """Run a simple performance test"""
    print_test("Performance Test")
    
    payload = {
        "question": "What are the key steps in data analysis?",
        "session_id": TEST_SESSION_ID,
        "chat_history": []
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            response_time = end_time - start_time
            print_success(f"Chat response time: {response_time:.2f} seconds")
            
            if response_time < 10:
                print_success("Performance is good (< 10 seconds)")
                return True
            elif response_time < 20:
                print_info("Performance is acceptable (< 20 seconds)")
                return True
            else:
                print_error("Performance is slow (> 20 seconds)")
                return False
        else:
            print_error("Performance test failed - endpoint not working")
            return False
            
    except requests.exceptions.RequestException as e:
        print_error(f"Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print_header("DataAnalystBot API Test Suite")
    print_info("Starting comprehensive API testing...")
    
    tests = [
        ("Server Health", test_server_health),
        ("Core Modules", test_core_modules),
        ("Chat Endpoint", test_chat_endpoint),
        ("CSV Upload", test_csv_upload_endpoint),
        ("Image Upload", test_image_upload_endpoint),
        ("Multi-Upload", test_multi_upload_endpoint),
        ("Data Analysis", test_data_analysis_endpoint),
        ("Data Cleaning", test_clean_data_endpoint),
        ("Session Management", test_session_management),
        ("Performance", run_performance_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Print summary
    print_header("Test Results Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:<8} {test_name}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Overall Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! Your DataAnalystBot is fully functional!{Colors.END}")
    elif passed >= total * 0.8:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Most tests passed. Some minor issues detected.{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Multiple tests failed. Please check the configuration.{Colors.END}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)