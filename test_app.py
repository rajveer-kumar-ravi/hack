#!/usr/bin/env python3
"""
Test script to verify the Fraud Detection API is working correctly
"""

import requests
import json
import time

def test_backend_api():
    """Test all backend API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Fraud Detection API...")
    print("=" * 50)
    
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
    
    # Test 3: Batch Analysis (with better sample data)
    print("\n3. Testing Batch Analysis...")
    try:
        # Create better sample CSV data with more records
        sample_csv = """Transaction_ID,User_ID,Transaction_Amount,Transaction_Type,Merchant_Category,Location,Time_of_Day,Day_of_Week,Transaction_Frequency,Average_Transaction_Amount,Account_Age,Risk_Score,Fraudulent
TX001,USER001,100.50,CASH_IN,FOOD_AND_DRINK,New York,14:30,Monday,5,85.20,365,0.1,0
TX002,USER002,5000.00,TRANSFER,OTHER,Los Angeles,09:15,Tuesday,2,2500.00,180,0.8,1
TX003,USER003,25.75,PAYMENT,SHOPPING,Chicago,16:45,Wednesday,8,30.00,730,0.2,0
TX004,USER004,15000.00,CASH_OUT,OTHER,Miami,23:30,Thursday,1,15000.00,90,0.9,1
TX005,USER005,45.20,DEBIT,TRANSPORT,Boston,07:20,Friday,12,40.00,1095,0.1,0
TX006,USER006,2500.00,TRANSFER,OTHER,Seattle,11:00,Saturday,3,1200.00,365,0.7,1
TX007,USER007,75.80,PAYMENT,ENTERTAINMENT,Denver,20:15,Sunday,6,65.00,540,0.3,0
TX008,USER008,8000.00,CASH_OUT,OTHER,Phoenix,02:45,Monday,1,8000.00,60,0.9,1
TX009,USER009,30.00,DEBIT,UTILITIES,Atlanta,10:30,Tuesday,15,35.00,1825,0.1,0
TX010,USER010,12000.00,TRANSFER,OTHER,San Francisco,18:20,Wednesday,2,6000.00,120,0.8,1"""
        
        files = {'file': ('test_data.csv', sample_csv, 'text/csv')}
        response = requests.post(f"{base_url}/api/batch-analysis", files=files)
        
        if response.status_code == 200:
            print("✅ Batch analysis passed")
            data = response.json()
            print(f"   Total Transactions: {data.get('statistics', {}).get('totalTransactions', 0)}")
            print(f"   Fraudulent Detected: {data.get('statistics', {}).get('fraudulentTransactions', 0)}")
            print(f"   Model Accuracy: {data.get('statistics', {}).get('accuracy', 0):.2%}")
            
            # Now test real-time analysis since model is trained
            print("\n4. Testing Real-time Analysis (after model training)...")
            try:
                transaction_data = {
                    "Transaction_ID": "TEST001",
                    "User_ID": "TESTUSER",
                    "Transaction_Amount": 5000,
                    "Transaction_Type": "TRANSFER",
                    "Merchant_Category": "OTHER",
                    "Location": "Test City",
                    "Time_of_Day": "15:30",
                    "Day_of_Week": "Monday",
                    "Transaction_Frequency": 5,
                    "Average_Transaction_Amount": 1000,
                    "Account_Age": 365,
                    "Risk_Score": 0.5
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
    
    print("\n" + "=" * 50)
    print("🎉 API Testing Complete!")

if __name__ == "__main__":
    test_backend_api()
