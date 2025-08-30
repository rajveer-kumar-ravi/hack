# Advanced Fraud Detection System

A comprehensive fraud detection system that combines supervised and unsupervised machine learning approaches to identify fraudulent transactions in real-time and batch processing modes.

## 🏗️ Project Structure

```
fraud-detection-system/
├── backend/                 # Python Flask API
│   ├── app.py              # Flask application with API endpoints
│   ├── backend.py          # Core ML algorithms and pipeline
│   ├── main.py             # Entry point for the backend server
│   ├── requirements.txt    # Python dependencies
│   └── __init__.py         # Package initialization
├── frontend/               # React.js frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Main application pages
│   │   ├── context/        # React context for state management
│   │   └── App.jsx         # Main React application
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- npm or yarn

### Backend Setup

1. **Create and activate virtual environment:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Flask backend:**
```bash
python main.py
```

The backend will start on `http://localhost:5000`

### Frontend Setup

1. **Install Node.js dependencies:**
```bash
cd frontend
npm install
```

2. **Start the React development server:**
```bash
npm start
```

The frontend will start on `http://localhost:3000`

## 🔧 Features

### Backend API Endpoints
- `GET /api/health` - Health check
- `GET /api/model-status` - Get model training status
- `POST /api/batch-analysis` - Upload CSV for batch fraud detection
- `POST /api/real-time-analysis` - Analyze single transaction

### Frontend Features
- **Dashboard** - Overview and statistics
- **Batch Analysis** - Upload and analyze CSV files
- **Real-time Analysis** - Analyze individual transactions
- **Results** - View detailed analysis results and flagged transactions

### Machine Learning Pipeline
1. **Supervised Learning** - Random Forest with class balancing
2. **Unsupervised Learning** - Isolation Forest for anomaly detection
3. **Ensemble Combination** - Weighted combination of supervised and unsupervised scores
4. **Threshold Tuning** - Automatic optimization of detection thresholds
5. **Graph-based Analysis** - Account-level fraud detection using transaction networks

## 📊 Data Format

The system expects CSV files with the following columns:
- `Transaction_ID` - Unique transaction identifier
- `User_ID` - User identifier
- `Transaction_Amount` - Transaction amount
- `Fraudulent` - Target column (0 for legitimate, 1 for fraudulent)
- Additional feature columns as needed

## 🛠️ Development

### Backend Development
```bash
cd backend
source venv/bin/activate
export FLASK_DEBUG=1
python main.py
```

### Frontend Development
```bash
cd frontend
npm start
```

## 📝 API Documentation

### Batch Analysis
```bash
curl -X POST http://localhost:5000/api/batch-analysis \
  -F "file=@transactions.csv" \
  -F "sample_size=10000"
```

### Real-time Analysis
```bash
curl -X POST http://localhost:5000/api/real-time-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "Transaction_ID": "TX001",
    "User_ID": "USER001", 
    "Transaction_Amount": 100.50
  }'
```

## 🔒 Security Features

- File upload size limits (configurable via `MAX_CONTENT_MB` environment variable)
- Input validation and sanitization
- CORS configuration for frontend integration
- Error handling and logging

## 📈 Performance

- Supports large datasets with configurable sampling
- Parallel processing for machine learning models
- Efficient memory management for large CSV files
- Real-time analysis with pre-trained models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. 