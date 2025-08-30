import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Shield, BarChart3, Zap, Home, Activity } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const Navbar = () => {
  const location = useLocation();
  const { modelStatus } = useFraudDetection();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/batch-analysis', label: 'Batch Analysis', icon: BarChart3 },
    { path: '/real-time', label: 'Real-Time', icon: Zap },
    { path: '/results', label: 'Results', icon: Activity },
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready':
        return 'text-success-600 bg-success-100';
      case 'training':
        return 'text-warning-600 bg-warning-100';
      case 'error':
        return 'text-danger-600 bg-danger-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <Shield className="h-8 w-8 text-primary-600" />
            <span className="text-xl font-bold text-gradient">FraudGuard</span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex space-x-8">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors duration-200 ${
                    isActive
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-600 hover:text-primary-600 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-medium">{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Model Status */}
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(modelStatus)}`}>
              Model: {modelStatus}
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="text-gray-600 hover:text-primary-600">
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 