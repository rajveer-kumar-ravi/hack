#!/bin/bash

echo "🚀 Starting Fraud Detection System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python3 -m venv venv' first."
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down services..."
    pkill -f "python main.py"
    pkill -f "npm start"
    pkill -f "react-scripts"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start backend
echo "🔧 Starting Python Flask backend..."
cd backend
source ../venv/bin/activate
# Set environment variables to reduce log spam
export STATUS_CHECK_COOLDOWN=60
export FLASK_DEBUG=0
echo "📊 Rate limiting: 60 seconds between status check logs"
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if ! curl -s http://localhost:5000/api/model-status > /dev/null; then
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
echo "✅ Backend is running on http://localhost:5000"

# Start frontend
echo "🌐 Starting React frontend..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 10

# Check if frontend is running
if ! curl -s http://localhost:3000 > /dev/null; then
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 1
fi
echo "✅ Frontend is running on http://localhost:3000"

echo ""
echo "🎉 Fraud Detection System is now running!"
echo "📊 Backend API: http://localhost:5000"
echo "🌐 Frontend App: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user to stop
wait
