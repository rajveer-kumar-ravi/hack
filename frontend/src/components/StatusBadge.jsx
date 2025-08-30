import React from 'react';
import { CheckCircle, AlertTriangle, Clock } from 'lucide-react';

const StatusBadge = ({ status, size = 'md' }) => {
  const statusConfig = {
    legitimate: {
      label: 'Legitimate',
      color: 'text-success-800',
      bgColor: 'bg-success-100',
      icon: CheckCircle
    },
    fraudulent: {
      label: 'Fraudulent',
      color: 'text-danger-800',
      bgColor: 'bg-danger-100',
      icon: AlertTriangle
    },
    suspicious: {
      label: 'Suspicious',
      color: 'text-warning-800',
      bgColor: 'bg-warning-100',
      icon: Clock
    }
  };

  const config = statusConfig[status] || statusConfig.suspicious;
  const Icon = config.icon;

  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-2.5 py-0.5 text-sm',
    lg: 'px-3 py-1 text-base'
  };

  return (
    <span className={`inline-flex items-center ${sizeClasses[size]} rounded-full font-medium ${config.bgColor} ${config.color}`}>
      <Icon className="h-3 w-3 mr-1" />
      {config.label}
    </span>
  );
};

export default StatusBadge; 