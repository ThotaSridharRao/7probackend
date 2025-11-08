#!/usr/bin/env python3
"""
Test DataAnalystBot with real datasets
"""

import requests
import base64
import pandas as pd
import os

API_BASE_URL = "http://localhost:8000"

def encode_csv_file(file_path):
    """Encode CSV file to base64"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def test_dataset(file_path, questions):
    """Test a dataset with multiple questions"""
    print(f"\n🧪 Testing dataset: {os.path.basename(file_path)}")
    print("=" * 50)
    
    # Load and show dataset info
    df = pd.read_csv(file_path)
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🔍 Sample data:")
    print(df.head(3).to_string())
    
    # Encode dataset
    csv_b64 = encode_csv_file(file_path)
    
    # Test each question
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        payload = {
            "question": question,
            "session_id": f"test_{os.path.basename(file_path)}_{i}",
            "chat_history": [],
            "csv_base64": csv_b64,
            "csv_filename": os.path.basename(file_path)
        }
        
        try:
            response = requests.post(f"{API_BASE_URL}/csv-upload", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('response', 'No response')
                print(f"✅ Answer: {answer[:200]}...")
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def test_data_analysis(file_path):
    """Test comprehensive data analysis"""
    print(f"\n🔬 Full Data Analysis: {os.path.basename(file_path)}")
    print("=" * 50)
    
    csv_b64 = encode_csv_file(file_path)
    
    payload = {
        "csv_base64": csv_b64,
        "csv_filename": os.path.basename(file_path),
        "session_id": f"analysis_{os.path.basename(file_path)}"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/analyze-data", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analysis completed: {data.get('success', False)}")
            print(f"📊 Original shape: {data.get('original_shape')}")
            print(f"🧹 Cleaned shape: {data.get('cleaned_shape')}")
            print(f"📝 Cleaning steps: {len(data.get('cleaning_log', []))}")
            print(f"📈 Visualizations: {len(data.get('visualizations', {}))}")
            
            # Show some insights
            insights = data.get('insights', '')
            if insights and len(insights) > 100:
                print(f"🤖 AI Insights preview: {insights[:300]}...")
            
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Test all sample datasets"""
    print("🚀 Testing DataAnalystBot with Sample Datasets")
    print("=" * 60)
    
    # Test datasets with specific questions
    datasets_and_questions = [
        ("sample_datasets/sales_data.csv", [
            "What are the top-selling product categories?",
            "Analyze sales trends by region",
            "What's the average sales amount by customer demographics?",
            "Which products have the highest quantity sold?"
        ]),
        
        ("sample_datasets/employee_data.csv", [
            "What's the salary distribution across departments?",
            "Analyze the relationship between experience and salary",
            "Which department has the highest performance scores?",
            "How does education level affect compensation?"
        ]),
        
        ("sample_datasets/customer_analytics.csv", [
            "Segment customers based on spending behavior",
            "What factors influence customer satisfaction?",
            "Analyze the relationship between age and spending score",
            "Which cities have the most loyal customers?"
        ])
    ]
    
    # Test each dataset
    for file_path, questions in datasets_and_questions:
        if os.path.exists(file_path):
            # Test with questions
            test_dataset(file_path, questions)
            
            # Test comprehensive analysis
            test_data_analysis(file_path)
            
            print("\n" + "="*60)
        else:
            print(f"❌ Dataset not found: {file_path}")
    
    print("\n🎉 Dataset testing completed!")
    print("\n💡 Try these online datasets next:")
    print("   • Kaggle: https://www.kaggle.com/datasets")
    print("   • UCI ML: https://archive.ics.uci.edu/ml/datasets.php")
    print("   • Data.gov: https://data.gov/")

if __name__ == "__main__":
    main()