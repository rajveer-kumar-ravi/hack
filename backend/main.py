#!/usr/bin/env python3
"""
Main entry point for the Fraud Detection API Server
"""

import os
import sys

# Add the current directory to Python path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app

if __name__ == '__main__':
    print("Starting Fraud Detection API Server...")
    print("Available endpoints:")
    print("- GET  /api/health - Health check")
    print("- GET  /api/model-status - Get model status")
    print("- POST /api/batch-analysis - Upload CSV for batch analysis")
    print("- POST /api/real-time-analysis - Analyze single transaction")
    print("- GET  /api/debug/rate-limit - Debug rate limiting")
    print("\nServer will start on http://localhost:5000")
    print(f"Status check cooldown: {os.environ.get('STATUS_CHECK_COOLDOWN', '30')} seconds")
    print("Rate limiting is ACTIVE to reduce log spam")
    
    # Respect environment variable to control debug to avoid dev server restarts causing request races
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
