#!/usr/bin/env python3
"""
Test the fixed endpoints
"""

import requests
import base64
import pandas as pd

def create_test_csv():
    """Create a test CSV file for testing"""
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000, 60000, 70000]
    }
    df = pd.DataFrame(data)
    csv_string = df.to_csv(index=False)
    return base64.b64encode(csv_string.encode()).decode()

def create_test_image():
    """Create a simple test image (1x1 pixel PNG)"""
    # Minimal PNG data for a 1x1 transparent pixel
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
    return base64.b64encode(png_data).decode()

def test_image_upload():
    """Test image upload"""
    print("🧪 Testing Image Upload...")
    
    image_data = create_test_image()
    payload = {
        "question": "What do you see in this image?",
        "session_id": "test_123",
        "chat_history": [],
        "image_base64": image_data,
        "image_type": "image/png"
    }
    
    try:
        response = requests.post("http://localhost:8000/image-upload", json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Image upload working!")
            print(f"Response: {data.get('response', 'No response')[:100]}...")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_data_cleaning():
    """Test data cleaning endpoint"""
    print("\n🧪 Testing Data Cleaning...")
    
    csv_data = create_test_csv()
    payload = {
        "csv_base64": csv_data,
        "csv_filename": "test.csv",
        "session_id": "test_123"
    }
    
    try:
        response = requests.post("http://localhost:8000/clean-data", json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Data cleaning working!")
            print(f"Success: {data.get('success', False)}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_image_upload()
    test_data_cleaning()