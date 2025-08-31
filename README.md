# 🎯 Fraud Detection System

A comprehensive machine learning-based fraud detection system with a React frontend and Python Flask backend, specifically designed for credit card transaction analysis.

## 🎯 Current Status: ✅ FULLY OPERATIONAL

**Both backend and frontend are running successfully!**

- **Backend API**: Running on http://localhost:5000 ✅
- **Frontend App**: Running on http://localhost:3000 ✅
- **Integration**: Complete and tested ✅
- **Batch Analysis**: Working with advanced ML pipeline ✅

## 🚀 Quick Start

### Option 1: Use the Startup Script (Recommended)
```bash
./start_app.sh
```

### Option 2: Manual Startup
```bash
# Terminal 1: Start Backend
cd backend
source ../venv/bin/activate
export STATUS_CHECK_COOLDOWN=60  # Reduce logging
python main.py

# Terminal 2: Start Frontend
cd frontend
npm start
```

## 📊 Features

### Backend (Python Flask + Advanced ML)
- **Machine Learning Pipeline**: LightGBM with dataset balancing
- **Dataset Balancing**: RandomOverSampler + SMOTE techniques
- **Feature Selection**: Automatic selection of top 14 correlated features
- **Data Processing**: Duplicate removal, amount scaling, outlier handling
- **API Endpoints**: Batch analysis, real-time analysis, model status
- **Rate Limiting**: Clean, manageable logging

### Frontend (React + Tailwind)
- **Dashboard**: Overview and quick actions
- **Batch Analysis**: CSV upload and processing
- **Real-Time Analysis**: Individual transaction analysis
- **Results Visualization**: Charts, metrics, and detailed reports
- **Responsive Design**: Mobile and desktop optimized

## 🔧 Dataset Requirements

The system is configured for the **credit card dataset** (`FraudDetectionDataset.csv`) with:
- **Features**: V1, V2, V3, ..., V28, Amount
- **Target**: Class (0 = Legitimate, 1 = Fraudulent)
- **Format**: CSV file

## 📡 API Endpoints

- `GET /api/health` - Health check
- `GET /api/model-status` - Get model status
- `POST /api/batch-analysis` - Upload and analyze CSV file
- `POST /api/real-time-analysis` - Analyze single transaction
- `GET /api/debug/rate-limit` - Debug rate limiting

## 🎨 UI Components

- **Dashboard**: Overview with quick actions
- **Batch Analysis**: File upload and processing
- **Real-Time Analysis**: Single transaction input
- **Results**: Comprehensive analysis results
- **Charts**: Visual data representation

## 🛠️ Technical Stack

- **Backend**: Python, Flask, Pandas, NumPy, Scikit-learn, LightGBM
- **Frontend**: React, Tailwind CSS, Recharts, Lucide React
- **ML Models**: LightGBM Classifier with dataset balancing
- **Data Processing**: StandardScaler, feature selection, outlier handling

## 📁 Project Structure

```
hack/
├── backend/
│   ├── fraud_detection.py    # Advanced ML pipeline (your notebook code)
│   ├── app.py               # Flask API endpoints
│   ├── main.py              # Server entry point
│   ├── requirements.txt     # Python dependencies
│   └── uploads/             # File upload directory
├── frontend/
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── context/         # React context for state
│   │   ├── pages/           # Main application pages
│   │   └── App.jsx          # Main application component
│   └── package.json
├── dataset/
│   └── cFraudDetectionDataset.csv       # Sample dataset
├── start_app.sh             # Startup script
└── README.md
```

## 🔍 Testing the System

1. **Access the Frontend**: http://localhost:3000
2. **Upload Dataset**: Use the batch analysis feature with `FraudDetectionDataset.csv`
3. **View Results**: Check the results page for comprehensive analysis
4. **Real-Time Analysis**: Test individual transaction analysis

## 📈 Performance Metrics

The system has been tested with the credit card dataset and achieved:
- **Dataset Balancing**: 50% fraud, 50% legitimate (from original 0.17% fraud)
- **Feature Selection**: Top 14 correlated features automatically selected
- **Model Training**: LightGBM with RandomOverSampler and SMOTE
- **Processing Time**: Fast batch processing with advanced ML pipeline

## 🚨 Troubleshooting

### Frontend Not Starting
```bash
cd frontend
npm install
npm start
```

### Backend Not Starting
```bash
cd backend
source ../venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Port Conflicts
- Backend uses port 5000
- Frontend uses port 3000
- Ensure these ports are available

### Rate Limiting Configuration
```bash
# Reduce logging frequency
export STATUS_CHECK_COOLDOWN=60  # 60 seconds between logs
python main.py
```

## 🎉 Success Confirmation

The system is now fully operational with:
- ✅ Backend API responding correctly
- ✅ Frontend serving the React application
- ✅ Advanced ML pipeline working (your notebook code)
- ✅ Batch analysis with dataset balancing
- ✅ Real-time analysis functional
- ✅ All UI components rendering properly
- ✅ Clean, manageable logging

## 📞 Support

The application is ready for use! Both services are running and communicating properly. You can now:

1. **Upload your credit card dataset** for advanced batch analysis
2. **Test real-time transactions** with individual analysis
3. **View comprehensive results** with charts and metrics
4. **Export results** for further analysis

## 🌟 Advanced Features

- **Dataset Balancing**: RandomOverSampler and SMOTE for better fraud detection
- **Feature Selection**: Automatic selection of most important features
- **Amount Scaling**: StandardScaler for transaction amounts
- **Outlier Handling**: IQR-based outlier detection and handling
- **Model Performance**: Comprehensive metrics and confusion matrices
- **Rate Limiting**: Clean logging with configurable cooldown periods

Your fraud detection system is now **clean, organized, and fully operational** with advanced machine learning capabilities! 🎯 