import React from 'react';
import { Loader } from 'lucide-react';

const LoadingSpinner = ({ size = 'md', text = 'Loading...' }) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
    xl: 'h-12 w-12'
  };

  return (
    <div className="flex items-center justify-center space-x-2">
      <Loader className={`${sizeClasses[size]} animate-spin text-primary-600`} />
      {text && <span className="text-gray-600">{text}</span>}
    </div>
  );
};

export default LoadingSpinner; 