#!/usr/bin/env python3
"""
Test script for the AIML Dataset
"""

import requests
import json
import time

def test_aiml_dataset():
    """Test the AIML dataset with the fraud detection API"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing AIML Dataset with Fraud Detection API...")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Model Status
    print("\n2. Testing Model Status...")
    try:
        response = requests.get(f"{base_url}/api/model-status")
        if response.status_code == 200:
            print("✅ Model status check passed")
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
        else:
            print(f"❌ Model status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model status error: {e}")
    
    # Test 3: Batch Analysis with AIML Dataset
    print("\n3. Testing Batch Analysis with AIML Dataset...")
    try:
        # Use the actual AIML dataset file
        dataset_path = "/home/harshita/Harshita/hack/dataset/AIML Dataset.csv"
        
        with open(dataset_path, 'rb') as f:
            files = {'file': ('AIML Dataset.csv', f, 'text/csv')}
            
            # Use a smaller sample size for testing (first 10000 rows)
            data = {'sample_size': '10000'}
            
            print(f"   Uploading dataset: {dataset_path}")
            print(f"   Sample size: 10,000 rows")
            
            response = requests.post(f"{base_url}/api/batch-analysis", files=files, data=data)
        
        if response.status_code == 200:
            print("✅ Batch analysis passed")
            data = response.json()
            print(f"   Total Transactions: {data.get('statistics', {}).get('totalTransactions', 0)}")
            print(f"   Fraudulent Detected: {data.get('statistics', {}).get('fraudulentTransactions', 0)}")
            print(f"   Legitimate Transactions: {data.get('statistics', {}).get('legitimateTransactions', 0)}")
            print(f"   Model Accuracy: {data.get('statistics', {}).get('accuracy', 0):.2%}")
            print(f"   Precision: {data.get('statistics', {}).get('precision', 0):.2%}")
            print(f"   Recall: {data.get('statistics', {}).get('recall', 0):.2%}")
            print(f"   F1 Score: {data.get('statistics', {}).get('f1Score', 0):.2%}")
            
            # Test real-time analysis with sample data from the dataset
            print("\n4. Testing Real-time Analysis...")
            try:
                transaction_data = {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 1000.0,
                    "nameOrig": "C123456789",
                    "oldbalanceOrg": 5000.0,
                    "newbalanceOrig": 4000.0,
                    "nameDest": "C987654321",
                    "oldbalanceDest": 1000.0,
                    "newbalanceDest": 2000.0,
                    "isFlaggedFraud": 0
                }
                
                response = requests.post(
                    f"{base_url}/api/real-time-analysis", 
                    json=transaction_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    print("✅ Real-time analysis passed")
                    data = response.json()
                    print(f"   Risk Level: {data.get('risk_level')}")
                    print(f"   Fraud Probability: {data.get('fraud_probability', 0):.2%}")
                    print(f"   Recommendation: {data.get('recommendation')}")
                else:
                    print(f"❌ Real-time analysis failed: {response.status_code}")
                    print(f"   Error: {response.text}")
            except Exception as e:
                print(f"❌ Real-time analysis error: {e}")
                
        else:
            print(f"❌ Batch analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 AIML Dataset Testing Complete!")

if __name__ == "__main__":
    test_aiml_dataset()
