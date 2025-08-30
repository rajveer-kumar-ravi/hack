#!/bin/bash

# Fraud Detection System - Complete Startup Script
echo "🚀 Starting Fraud Detection System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down servers...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo -e "${RED}❌ Error: Please run this script from the project root directory${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [ -z "$python_version" ]; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python $python_version found${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ npm found${NC}"

echo -e "\n${BLUE}🔧 Setting up Backend...${NC}"

# Setup backend
cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}🔌 Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}📥 Installing Python dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1

# Install requests for testing
pip install requests > /dev/null 2>&1

# Start backend server
echo -e "${YELLOW}🚀 Starting backend server...${NC}"
python main.py &
BACKEND_PID=$!

cd ..

# Wait for backend to start
echo -e "${YELLOW}⏳ Waiting for backend to start...${NC}"
sleep 5

# Test backend
if curl -s http://localhost:5000/api/health > /dev/null; then
    echo -e "${GREEN}✅ Backend server running on http://localhost:5000${NC}"
else
    echo -e "${RED}❌ Backend failed to start${NC}"
    exit 1
fi

echo -e "\n${BLUE}🎨 Setting up Frontend...${NC}"

# Setup frontend
cd frontend

# Install dependencies
echo -e "${YELLOW}📥 Installing Node.js dependencies...${NC}"
npm install > /dev/null 2>&1

# Start frontend server
echo -e "${YELLOW}🚀 Starting frontend server...${NC}"
npm start &
FRONTEND_PID=$!

cd ..

# Wait for frontend to start
echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
sleep 10

# Test frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ Frontend server running on http://localhost:3000${NC}"
else
    echo -e "${YELLOW}⚠️  Frontend may still be starting...${NC}"
fi

echo -e "\n${GREEN}🎉 Fraud Detection System is ready!${NC}"
echo -e "${BLUE}📊 Frontend: http://localhost:3000${NC}"
echo -e "${BLUE}🔧 Backend API: http://localhost:5000${NC}"
echo -e "${BLUE}📋 Health Check: http://localhost:5000/api/health${NC}"
echo -e "\n${YELLOW}💡 Tips:${NC}"
echo -e "   • Upload the sample_data.csv file for testing"
echo -e "   • Use the Real-Time Analysis for individual transactions"
echo -e "   • Check the Results page for detailed analytics"
echo -e "\n${YELLOW}Press Ctrl+C to stop all servers${NC}"

# Wait for both processes
wait
