import React from 'react';
import { CheckCircle, Clock, AlertTriangle, Activity } from 'lucide-react';

const StatusBadge = ({ status, className = '', size = 'md' }) => {
  const getStatusConfig = (status) => {
    switch (status) {
      case 'ready':
        return {
          text: 'Ready',
          color: 'text-green-600',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          icon: CheckCircle
        };
      case 'training':
        return {
          text: 'Training',
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-50',
          borderColor: 'border-yellow-200',
          icon: Activity
        };
      case 'error':
        return {
          text: 'Error',
          color: 'text-red-600',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          icon: AlertTriangle
        };
      case 'idle':
      default:
        return {
          text: 'Idle',
          color: 'text-gray-600',
          bgColor: 'bg-gray-50',
          borderColor: 'border-gray-200',
          icon: Clock
        };
    }
  };

  const getSizeClasses = (size) => {
    switch (size) {
      case 'sm':
        return {
          container: 'px-2 py-1 text-xs',
          icon: 'h-3 w-3 sm:h-4 sm:w-4 mr-1 sm:mr-2'
        };
      case 'lg':
        return {
          container: 'px-4 py-2 text-base',
          icon: 'h-5 w-5 sm:h-6 sm:w-6 mr-2 sm:mr-3'
        };
      case 'md':
      default:
        return {
          container: 'px-3 py-1 text-sm',
          icon: 'h-4 w-4 mr-2'
        };
    }
  };

  const config = getStatusConfig(status);
  const sizeClasses = getSizeClasses(size);
  const Icon = config.icon;

  return (
    <div className={`inline-flex items-center rounded-full font-medium border ${config.bgColor} ${config.borderColor} ${config.color} ${sizeClasses.container} ${className}`}>
      <Icon className={sizeClasses.icon} />
      <span className="text-responsive-sm">{config.text}</span>
    </div>
  );
};

export default StatusBadge; 