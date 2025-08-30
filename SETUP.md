# 🚀 Fraud Detection System - Complete Setup Guide

## ✅ Project Reorganization Complete!

The project has been successfully reorganized with a clean structure:

```
fraud-detection-system/
├── backend/                 # Python Flask API
│   ├── app.py              # Flask application with API endpoints
│   ├── backend.py          # Core ML algorithms and pipeline
│   ├── main.py             # Entry point for the backend server
│   ├── requirements.txt    # Python dependencies
│   ├── __init__.py         # Package initialization
│   └── venv/               # Virtual environment (created during setup)
├── frontend/               # React.js frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Main application pages
│   │   ├── context/        # React context for state management
│   │   └── App.jsx         # Main React application
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
├── run.sh                  # One-click startup script
├── README.md               # Project documentation
├── SETUP.md                # This setup guide
└── .gitignore              # Git ignore rules
```

## 🎯 Quick Start (3 Options)

### Option 1: One-Click Startup (Recommended)
```bash
./run.sh
```
This script will:
- Set up the Python virtual environment
- Install all dependencies
- Start both backend and frontend servers
- Open the application in your browser

### Option 2: Manual Setup

#### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

#### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

### Option 3: Individual Terminal Sessions

#### Terminal 1 - Backend
```bash
cd backend
source venv/bin/activate
python main.py
```

#### Terminal 2 - Frontend
```bash
cd frontend
npm start
```

## 🌐 Access Points

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health

## 🔧 API Endpoints

### Health & Status
- `GET /api/health` - Server health check
- `GET /api/model-status` - Model training status

### Analysis
- `POST /api/batch-analysis` - Upload CSV for batch analysis
- `POST /api/real-time-analysis` - Analyze single transaction

## 📊 Features

### Backend (Python Flask + ML)
- ✅ Supervised Learning (Random Forest)
- ✅ Unsupervised Learning (Isolation Forest)
- ✅ Hybrid Ensemble Approach
- ✅ Automatic Threshold Tuning
- ✅ Graph-based Account Analysis
- ✅ RESTful API with CORS
- ✅ File Upload Support
- ✅ Real-time Processing

### Frontend (React + Tailwind)
- ✅ Modern Dashboard
- ✅ Batch Analysis Interface
- ✅ Real-time Analysis Form
- ✅ Results Visualization
- ✅ Responsive Design
- ✅ Interactive Charts
- ✅ File Upload with Drag & Drop

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

## 📝 Testing the Setup

### 1. Test Backend Health
```bash
curl http://localhost:5000/api/health
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": false
}
```

### 2. Test Frontend
Open http://localhost:3000 in your browser
- Should see the dashboard
- Navigation should work
- No console errors

### 3. Test File Upload
- Navigate to "Batch Analysis"
- Upload a CSV file with transaction data
- Should process and show results

## 🔍 Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Reinstall dependencies
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Frontend Won't Start
```bash
# Clear npm cache
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### Port Already in Use
```bash
# Find and kill processes
lsof -ti:5000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend
```

### Environment Variables

#### Backend
- `FLASK_DEBUG=1` - Enable debug mode
- `MAX_CONTENT_MB=2048` - Max file upload size (MB)

#### Frontend
- `REACT_APP_API_URL=http://localhost:5000` - Backend API URL

## 📈 Performance Tips

1. **Large Datasets**: Use the `sample_size` parameter in batch analysis
2. **Memory**: Monitor system resources during large file processing
3. **Caching**: Models are cached after first training for faster subsequent runs

## 🔒 Security Notes

- File uploads are validated and sanitized
- CORS is configured for localhost development
- Temporary files are automatically cleaned up
- Input validation on all API endpoints

## 📞 Support

If you encounter issues:

1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure ports 3000 and 5000 are available
4. Check the troubleshooting section above

## 🎉 Success!

Your Fraud Detection System is now ready to use! 

- Upload transaction data for batch analysis
- Use real-time analysis for individual transactions
- View detailed statistics and flagged transactions
- Export results for further analysis

Happy fraud detecting! 🛡️
