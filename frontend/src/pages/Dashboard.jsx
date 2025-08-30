import React from 'react';
import { Link } from 'react-router-dom';
import { 
  BarChart3, 
  Zap, 
  Shield, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Activity
} from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const Dashboard = () => {
  const { modelStatus, batchResults, realTimeResults } = useFraudDetection();

  const stats = [
    {
      title: 'Total Transactions',
      value: batchResults?.statistics?.totalTransactions || 0,
      icon: Activity,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      title: 'Fraudulent Detected',
      value: batchResults?.statistics?.fraudulentTransactions || 0,
      icon: AlertTriangle,
      color: 'text-danger-600',
      bgColor: 'bg-danger-50'
    },
    {
      title: 'Legitimate Transactions',
      value: batchResults?.statistics?.legitimateTransactions || 0,
      icon: CheckCircle,
      color: 'text-success-600',
      bgColor: 'bg-success-50'
    },
    {
      title: 'Model Accuracy',
      value: `${((batchResults?.statistics?.accuracy || 0) * 100).toFixed(2)}%`,
      icon: TrendingUp,
      color: 'text-warning-600',
      bgColor: 'bg-warning-50'
    }
  ];

  const quickActions = [
    {
      title: 'Batch Analysis',
      description: 'Upload CSV file for bulk fraud detection',
      icon: BarChart3,
      path: '/batch-analysis',
      color: 'bg-primary-500 hover:bg-primary-600'
    },
    {
      title: 'Real-Time Analysis',
      description: 'Analyze individual transactions instantly',
      icon: Zap,
      path: '/real-time',
      color: 'bg-success-500 hover:bg-success-600'
    },
    {
      title: 'View Results',
      description: 'Check detailed analysis results',
      icon: Shield,
      path: '/results',
      color: 'bg-warning-500 hover:bg-warning-600'
    }
  ];

  const getModelStatusInfo = () => {
    switch (modelStatus) {
      case 'ready':
        return {
          text: 'Model Ready',
          color: 'text-success-600',
          bgColor: 'bg-success-50',
          icon: CheckCircle
        };
      case 'training':
        return {
          text: 'Training in Progress',
          color: 'text-warning-600',
          bgColor: 'bg-warning-50',
          icon: Clock
        };
      case 'error':
        return {
          text: 'Model Error',
          color: 'text-danger-600',
          bgColor: 'bg-danger-50',
          icon: AlertTriangle
        };
      default:
        return {
          text: 'Initializing',
          color: 'text-gray-600',
          bgColor: 'bg-gray-50',
          icon: Clock
        };
    }
  };

  const statusInfo = getModelStatusInfo();
  const StatusIcon = statusInfo.icon;

  return (
    <div className="space-y-6 sm:space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-3 sm:mb-4">
          Fraud Detection Dashboard
        </h1>
        <p className="text-base sm:text-lg text-gray-600 max-w-2xl mx-auto px-4">
          Advanced machine learning system for detecting fraudulent transactions using supervised and unsupervised learning techniques.
        </p>
      </div>

      {/* Model Status */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${statusInfo.bgColor}`}>
              <StatusIcon className={`h-5 w-5 sm:h-6 sm:w-6 ${statusInfo.color}`} />
            </div>
            <div>
              <h3 className="text-base sm:text-lg font-semibold text-gray-900">Model Status</h3>
              <p className={`text-sm font-medium ${statusInfo.color}`}>
                {statusInfo.text}
              </p>
            </div>
          </div>
          <div className="text-left sm:text-right">
            <p className="text-sm text-gray-500">Last Updated</p>
            <p className="text-sm font-medium text-gray-900">
              {new Date().toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="card">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-xs sm:text-sm font-medium text-gray-600 truncate">{stat.title}</p>
                  <p className="text-xl sm:text-2xl font-bold text-gray-900 truncate">{stat.value}</p>
                </div>
                <div className={`p-2 sm:p-3 rounded-lg flex-shrink-0 ${stat.bgColor}`}>
                  <Icon className={`h-5 w-5 sm:h-6 sm:w-6 ${stat.color}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {quickActions.map((action, index) => {
            const Icon = action.icon;
            return (
              <Link
                key={index}
                to={action.path}
                className="group block p-4 sm:p-6 border border-gray-200 rounded-xl hover:border-primary-300 hover:shadow-lg transition-all duration-200"
              >
                <div className="flex items-center space-x-3 sm:space-x-4">
                  <div className={`p-2 sm:p-3 rounded-lg ${action.color} text-white transition-colors duration-200 flex-shrink-0`}>
                    <Icon className="h-5 w-5 sm:h-6 sm:w-6" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-base sm:text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors duration-200 truncate">
                      {action.title}
                    </h3>
                    <p className="text-xs sm:text-sm text-gray-600 mt-1 line-clamp-2">
                      {action.description}
                    </p>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Recent Activity */}
      {(batchResults || realTimeResults) && (
        <div className="card">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Recent Activity</h2>
          <div className="space-y-3 sm:space-y-4">
            {batchResults && (
              <div className="flex items-center space-x-3 p-3 sm:p-4 bg-blue-50 rounded-lg">
                <BarChart3 className="h-4 w-4 sm:h-5 sm:w-5 text-blue-600 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 text-sm sm:text-base">Batch Analysis Completed</p>
                  <p className="text-xs sm:text-sm text-gray-600 truncate">
                    {batchResults.statistics?.totalTransactions || 0} transactions analyzed
                  </p>
                </div>
              </div>
            )}
            {realTimeResults && (
              <div className="flex items-center space-x-3 p-3 sm:p-4 bg-green-50 rounded-lg">
                <Zap className="h-4 w-4 sm:h-5 sm:w-5 text-green-600 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 text-sm sm:text-base">Real-Time Analysis</p>
                  <p className="text-xs sm:text-sm text-gray-600 truncate">
                    Transaction analyzed with {realTimeResults.confidence || 0}% confidence
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard; 