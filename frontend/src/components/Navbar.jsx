import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Shield, BarChart3, Zap, Home, Activity, Menu, X } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const Navbar = () => {
  const location = useLocation();
  const { modelStatus } = useFraudDetection();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  return (
    <nav className="bg-white shadow-lg border-b border-gray-200 relative z-50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2" onClick={closeMobileMenu}>
            <Shield className="h-6 w-6 sm:h-8 sm:w-8 text-primary-600" />
            <span className="text-lg sm:text-xl font-bold text-gradient">FraudLens</span>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex space-x-4 lg:space-x-8">
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
                  <Icon className="h-4 w-4 lg:h-5 lg:w-5" />
                  <span className="font-medium text-sm lg:text-base">{item.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Model Status - Desktop */}
          <div className="hidden md:flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(modelStatus)}`}>
              <span className="hidden lg:inline">Model: </span>{modelStatus}
            </div>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center space-x-2">
            {/* Model Status - Mobile */}
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(modelStatus)}`}>
              {modelStatus}
            </div>
            
            <button 
              onClick={toggleMobileMenu}
              className="text-gray-600 hover:text-primary-600 p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200"
              aria-label="Toggle mobile menu"
            >
              {isMobileMenuOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div className={`md:hidden transition-all duration-300 ease-in-out ${
          isMobileMenuOpen 
            ? 'max-h-96 opacity-100 visible' 
            : 'max-h-0 opacity-0 invisible'
        }`}>
          <div className="py-4 space-y-2 border-t border-gray-200">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={closeMobileMenu}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors duration-200 ${
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
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 