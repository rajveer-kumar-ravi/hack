import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import BatchAnalysis from './pages/BatchAnalysis';
import RealTimeAnalysis from './pages/RealTimeAnalysis';
import Results from './pages/Results';
import { FraudDetectionProvider } from './context/FraudDetectionContext';

function App() {
  return (
    <FraudDetectionProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Navbar />
          <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6 lg:py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/batch-analysis" element={<BatchAnalysis />} />
              <Route path="/real-time" element={<RealTimeAnalysis />} />
              <Route path="/results" element={<Results />} />
            </Routes>
          </main>
        </div>
      </Router>
    </FraudDetectionProvider>
  );
}

export default App; 