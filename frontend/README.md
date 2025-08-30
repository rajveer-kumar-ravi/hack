# Fraud Detection System Frontend

A modern React-based frontend for the advanced fraud detection system using machine learning techniques.

## Features

- **Dashboard**: Overview of system status and key metrics
- **Batch Analysis**: Upload CSV files for bulk fraud detection
- **Real-Time Analysis**: Analyze individual transactions instantly
- **Results Visualization**: Detailed charts and tables for analysis results
- **Modern UI**: Built with React, Tailwind CSS, and Lucide React icons

## Tech Stack

- **React 18**: Modern React with hooks and context
- **Tailwind CSS**: Utility-first CSS framework
- **React Router**: Client-side routing
- **Axios**: HTTP client for API calls
- **Recharts**: Data visualization library
- **Lucide React**: Beautiful icons
- **React Dropzone**: File upload functionality

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Python backend running on localhost:5000

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd hack/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

The application will open at `http://localhost:3000`

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Navbar.js
│   │   ├── LoadingSpinner.js
│   │   └── StatusBadge.js
│   ├── context/
│   │   └── FraudDetectionContext.js
│   ├── pages/
│   │   ├── Dashboard.js
│   │   ├── BatchAnalysis.js
│   │   ├── RealTimeAnalysis.js
│   │   └── Results.js
│   ├── App.js
│   ├── index.js
│   └── index.css
├── package.json
├── tailwind.config.js
└── postcss.config.js
```

## Usage

### Dashboard
- View system overview and statistics
- Quick access to all features
- Real-time model status

### Batch Analysis
1. Navigate to "Batch Analysis"
2. Drag and drop or select a CSV file
3. Optionally set sample size for faster processing
4. Click "Start Analysis"
5. View results and download flagged transactions

### Real-Time Analysis
1. Navigate to "Real-Time Analysis"
2. Fill in transaction details
3. Click "Analyze Transaction"
4. View risk assessment and recommendations

### Results
- View detailed analysis results
- Interactive charts and visualizations
- Export results to CSV
- Browse flagged transactions

## API Integration

The frontend expects the following API endpoints from the Python backend:

- `GET /api/model-status` - Get model status
- `POST /api/batch-analysis` - Upload and analyze CSV file
- `POST /api/real-time-analysis` - Analyze single transaction

## Configuration

### Backend URL
The frontend is configured to connect to `http://localhost:5000` by default. To change this:

1. Update the `baseURL` in `src/context/FraudDetectionContext.js`
2. Or set the `REACT_APP_API_URL` environment variable

### Styling
Customize the design by modifying:
- `tailwind.config.js` for theme configuration
- `src/index.css` for custom styles
- Component-specific classes

## Development

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App

### Code Style

- Use functional components with hooks
- Follow React best practices
- Use Tailwind CSS for styling
- Maintain consistent component structure

## Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure Python backend is running on port 5000
   - Check CORS configuration in backend

2. **File Upload Issues**
   - Ensure CSV file format is correct
   - Check file size limits

3. **Build Errors**
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility

## Contributing

1. Follow the existing code style
2. Add proper error handling
3. Test thoroughly before submitting
4. Update documentation as needed

## License

This project is part of the fraud detection system. 