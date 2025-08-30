# Advanced Fraud Detection System

A comprehensive fraud detection system with a modern React frontend and Python Flask backend, featuring machine learning algorithms for detecting fraudulent transactions.

## 🚀 Features

### Frontend (React + Tailwind CSS)
- **Modern Dashboard**: Real-time system overview and statistics
- **Batch Analysis**: Upload CSV files for bulk fraud detection
- **Real-Time Analysis**: Analyze individual transactions instantly
- **Interactive Results**: Charts, tables, and visualizations
- **Responsive Design**: Works on desktop and mobile devices

### Backend (Python Flask + ML)
- **Supervised Learning**: Random Forest classifier with balanced sampling
- **Unsupervised Learning**: Isolation Forest for anomaly detection
- **Hybrid Approach**: Combined supervised and unsupervised scores
- **Threshold Optimization**: Automatic threshold tuning for optimal F1 score
- **Graph Analysis**: GNN-baseline for account-level flagging
- **REST API**: Full API integration with the frontend

## 🛠️ Tech Stack

### Frontend
- React 18 with Hooks and Context
- Tailwind CSS for styling
- React Router for navigation
- Axios for API communication
- Recharts for data visualization
- Lucide React for icons
- React Dropzone for file uploads

### Backend
- Python 3.8+
- Flask for API server
- Scikit-learn for machine learning
- Pandas for data processing
- NumPy for numerical operations
- NetworkX for graph analysis

## 📋 Prerequisites

- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **npm** or **yarn**
- **pip** (Python package manager)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Navigate to the project directory
cd hack

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Start the Backend Server

```bash
# Start the Flask API server
python app.py
```

The backend will start on `http://localhost:5000`

### 3. Start the Frontend

```bash
# In a new terminal, start the React development server
cd frontend
npm start
```

The frontend will open at `http://localhost:3000`

## 📁 Project Structure

```
hack/
├── backend.py              # Original fraud detection pipeline
├── app.py                  # Flask API server
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── frontend/              # React frontend application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── context/       # React context for state management
│   │   ├── pages/         # Main application pages
│   │   ├── App.js         # Main app component
│   │   ├── index.js       # React entry point
│   │   └── index.css      # Global styles
│   ├── package.json       # Node.js dependencies
│   ├── tailwind.config.js # Tailwind CSS configuration
│   └── README.md          # Frontend documentation
└── uploads/               # Temporary file upload directory
```

## 🔧 Configuration

### Backend Configuration

The backend can be configured by modifying variables in `app.py`:

```python
# Target column for fraud detection
TARGET_COL = 'Fraudulent'

# ID columns for transaction identification
ID_COLS = ['Transaction_ID', 'User_ID']

# File upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
```

### Frontend Configuration

The frontend connects to the backend via the proxy configuration in `package.json`:

```json
{
  "proxy": "http://localhost:5000"
}
```

## 📊 API Endpoints

### Health Check
- `GET /api/health` - Check server status

### Model Status
- `GET /api/model-status` - Get current model status

### Batch Analysis
- `POST /api/batch-analysis` - Upload CSV for bulk analysis
  - Form data: `file` (CSV), `sample_size` (optional)

### Real-Time Analysis
- `POST /api/real-time-analysis` - Analyze single transaction
  - JSON body: Transaction data

## 🎯 Usage Guide

### 1. Dashboard
- View system overview and statistics
- Check model status
- Quick access to all features

### 2. Batch Analysis
1. Navigate to "Batch Analysis"
2. Upload a CSV file with transaction data
3. Optionally set sample size for faster processing
4. Click "Start Analysis"
5. View comprehensive results and download flagged transactions

### 3. Real-Time Analysis
1. Navigate to "Real-Time Analysis"
2. Fill in transaction details in the form
3. Click "Analyze Transaction"
4. View risk assessment and recommendations

### 4. Results
- Interactive charts and visualizations
- Detailed statistics and metrics
- Export functionality for flagged transactions
- Browse and filter results

## 📈 Data Format

### Expected CSV Columns
Your CSV file should include these columns:
- `Transaction_ID` - Unique transaction identifier
- `User_ID` - User account identifier
- `Transaction_Amount` - Transaction amount
- `Fraudulent` - Target column (0 for legitimate, 1 for fraudulent)
- Additional features for better detection

### Example CSV Structure
```csv
Transaction_ID,User_ID,Transaction_Amount,Transaction_Type,Merchant_Category,Fraudulent
TX001,USER001,100.50,CASH_IN,FOOD_AND_DRINK,0
TX002,USER002,5000.00,TRANSFER,OTHER,1
```

## 🔍 Machine Learning Pipeline

The system uses a sophisticated multi-stage approach:

1. **Data Preprocessing**: Label encoding, scaling, and cleaning
2. **Supervised Learning**: Random Forest with balanced sampling
3. **Unsupervised Learning**: Isolation Forest for anomaly detection
4. **Score Combination**: Weighted combination of both approaches
5. **Threshold Optimization**: Automatic tuning for optimal performance
6. **Graph Analysis**: Account-level flagging using GNN-baseline

## 🚨 Troubleshooting

### Common Issues

1. **Backend Connection Error**
   ```bash
   # Check if backend is running
   curl http://localhost:5000/api/health
   ```

2. **CORS Issues**
   - Ensure Flask-CORS is installed: `pip install flask-cors`
   - Check that CORS is enabled in `app.py`

3. **File Upload Issues**
   - Check file size (max 16MB)
   - Ensure CSV format is correct
   - Verify required columns are present

4. **Model Training Issues**
   - Check data format and required columns
   - Ensure sufficient data for training
   - Verify target column exists

### Debug Mode

Enable debug mode for detailed error messages:

```python
# In app.py
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🔒 Security Considerations

- File upload validation and sanitization
- CORS configuration for frontend-backend communication
- Input validation for all API endpoints
- Temporary file cleanup after processing

## 📝 Development

### Adding New Features

1. **Backend**: Add new endpoints in `app.py`
2. **Frontend**: Create new components in `frontend/src/components/`
3. **Pages**: Add new pages in `frontend/src/pages/`
4. **Styling**: Use Tailwind CSS classes or add custom styles

### Testing

```bash
# Test backend
python -m pytest tests/

# Test frontend
cd frontend
npm test
```

## 📄 License

This project is part of the fraud detection system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check the console for error messages
4. Verify data format and requirements

---

**Happy Fraud Detection! 🛡️** 