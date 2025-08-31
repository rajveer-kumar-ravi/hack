import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { Download, Eye, Filter, TrendingUp, AlertTriangle, CheckCircle, X, Calendar, DollarSign, User, Hash } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const Results = () => {
  const { batchResults, realTimeResults } = useFraudDetection();
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const COLORS = ['#10B981', '#F59E0B', '#EF4444'];

  const prepareChartData = () => {
    if (!batchResults?.statistics) return [];
    
    return [
      { name: 'Legitimate', value: batchResults.statistics.legitimateTransactions, color: '#10B981' },
      { name: 'Fraudulent', value: batchResults.statistics.fraudulentTransactions, color: '#EF4444' }
    ];
  };

  const prepareMetricsData = () => {
    if (!batchResults?.statistics) return [];
    
    return [
      { name: 'Accuracy', value: batchResults.statistics.accuracy * 100 },
      { name: 'Precision', value: batchResults.statistics.precision * 100 },
      { name: 'Recall', value: batchResults.statistics.recall * 100 },
      { name: 'F1 Score', value: batchResults.statistics.f1Score * 100 }
    ];
  };

  const exportResults = () => {
    if (!batchResults?.flaggedTransactions) return;
    
    const csvContent = "data:text/csv;charset=utf-8," + 
      "Transaction_ID,User_ID,Transaction_Amount,Suspicion_Score,Status\n" +
      batchResults.flaggedTransactions.map(tx => 
        `${tx.Transaction_ID || tx.transaction_id || 'N/A'},${tx.User_ID || tx.user_id || 'N/A'},${tx.Transaction_Amount || tx.amount || 0},${tx.suspicion_score || 0},Fraudulent`
      ).join('\n');
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "fraud_detection_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleViewTransaction = (transaction) => {
    setSelectedTransaction(transaction);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setSelectedTransaction(null);
  };

  const getRiskLevel = (score) => {
    if (score >= 0.8) return { level: 'High Risk', color: 'text-danger-600', bgColor: 'bg-danger-50' };
    if (score >= 0.5) return { level: 'Medium Risk', color: 'text-warning-600', bgColor: 'bg-warning-50' };
    return { level: 'Low Risk', color: 'text-success-600', bgColor: 'bg-success-50' };
  };

  if (!batchResults && !realTimeResults) {
    return (
      <div className="max-w-4xl mx-auto text-center py-8 sm:py-12">
        <div className="card">
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-3 sm:mb-4">No Results Available</h1>
          <p className="text-base sm:text-lg text-gray-600 mb-6 sm:mb-8 px-4">
            Run a batch analysis or real-time analysis to see results here.
          </p>
          <div className="flex flex-col sm:flex-row justify-center space-y-3 sm:space-y-0 sm:space-x-4">
            <button className="btn-primary">Run Batch Analysis</button>
            <button className="btn-secondary">Real-Time Analysis</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6 sm:space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center space-y-4 sm:space-y-0">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-2">Analysis Results</h1>
          <p className="text-base sm:text-lg text-gray-600">
            Detailed insights from fraud detection analysis
          </p>
        </div>
        {batchResults && (
          <button
            onClick={exportResults}
            className="btn-primary flex items-center justify-center space-x-2 w-full sm:w-auto"
          >
            <Download className="h-4 w-4 sm:h-5 sm:w-5" />
            <span className="text-sm sm:text-base">Export Results</span>
          </button>
        )}
      </div>

      {/* Summary Cards */}
      {batchResults && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
          <div className="card">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <p className="text-xs sm:text-sm font-medium text-gray-600">Total Transactions</p>
                <p className="text-xl sm:text-2xl font-bold text-gray-900">
                  {batchResults.statistics?.totalTransactions || 0}
                </p>
              </div>
              <div className="p-2 sm:p-3 bg-blue-50 rounded-lg flex-shrink-0">
                <TrendingUp className="h-5 w-5 sm:h-6 sm:w-6 text-blue-600" />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <p className="text-xs sm:text-sm font-medium text-gray-600">Fraudulent</p>
                <p className="text-xl sm:text-2xl font-bold text-danger-600">
                  {batchResults.statistics?.fraudulentTransactions || 0}
                </p>
              </div>
              <div className="p-2 sm:p-3 bg-danger-50 rounded-lg flex-shrink-0">
                <AlertTriangle className="h-5 w-5 sm:h-6 sm:w-6 text-danger-600" />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <p className="text-xs sm:text-sm font-medium text-gray-600">Legitimate</p>
                <p className="text-xl sm:text-2xl font-bold text-success-600">
                  {batchResults.statistics?.legitimateTransactions || 0}
                </p>
              </div>
              <div className="p-2 sm:p-3 bg-success-50 rounded-lg flex-shrink-0">
                <CheckCircle className="h-5 w-5 sm:h-6 sm:w-6 text-success-600" />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div className="flex-1 min-w-0">
                <p className="text-xs sm:text-sm font-medium text-gray-600">Accuracy</p>
                <p className="text-xl sm:text-2xl font-bold text-warning-600">
                  {((batchResults.statistics?.accuracy || 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className="p-2 sm:p-3 bg-warning-50 rounded-lg flex-shrink-0">
                <TrendingUp className="h-5 w-5 sm:h-6 sm:w-6 text-warning-600" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      {batchResults && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
          {/* Transaction Distribution */}
          <div className="card">
            <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Transaction Distribution</h2>
            <div className="flex justify-center">
              <PieChart width={Math.min(400, window.innerWidth - 100)} height={300}>
                <Pie
                  data={prepareChartData()}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {prepareChartData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </div>
          </div>

          {/* Model Performance Metrics */}
          <div className="card">
            <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Model Performance</h2>
            <div className="flex justify-center">
              <BarChart width={Math.min(400, window.innerWidth - 100)} height={300} data={prepareMetricsData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3B82F6" />
              </BarChart>
            </div>
          </div>
        </div>
      )}

      {/* Confusion Matrix Summary */}
      {batchResults?.statistics && (
        <div className="card">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Confusion Matrix Summary</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-success-50 rounded-lg">
              <p className="text-2xl font-bold text-success-600">{batchResults.statistics.trueNegatives || 0}</p>
              <p className="text-sm text-gray-600">True Negatives</p>
            </div>
            <div className="text-center p-4 bg-danger-50 rounded-lg">
              <p className="text-2xl font-bold text-danger-600">{batchResults.statistics.falsePositives || 0}</p>
              <p className="text-sm text-gray-600">False Positives</p>
            </div>
            <div className="text-center p-4 bg-warning-50 rounded-lg">
              <p className="text-2xl font-bold text-warning-600">{batchResults.statistics.falseNegatives || 0}</p>
              <p className="text-sm text-gray-600">False Negatives</p>
            </div>
            <div className="text-center p-4 bg-primary-50 rounded-lg">
              <p className="text-2xl font-bold text-primary-600">{batchResults.statistics.truePositives || 0}</p>
              <p className="text-sm text-gray-600">True Positives</p>
            </div>
          </div>
        </div>
      )}

      {/* Real-Time Results */}
      {realTimeResults && (
        <div className="card">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Real-Time Analysis Result</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
            <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2 text-sm sm:text-base">Fraud Probability</h3>
              <p className="text-2xl sm:text-3xl font-bold text-danger-600">
                {(realTimeResults.fraud_probability * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2 text-sm sm:text-base">Confidence</h3>
              <p className="text-2xl sm:text-3xl font-bold text-primary-600">
                {(realTimeResults.confidence * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-3 sm:p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2 text-sm sm:text-base">Risk Level</h3>
              <p className="text-xl sm:text-2xl font-bold text-warning-600">
                {realTimeResults.risk_level || 'Unknown'}
              </p>
            </div>
          </div>
          
          {/* Detailed Scores */}
          <div className="mt-4 sm:mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2 text-sm">Supervised Score</h4>
              <p className="text-lg font-bold text-primary-600">
                {(realTimeResults.supervised_score * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2 text-sm">Anomaly Score</h4>
              <p className="text-lg font-bold text-warning-600">
                {(realTimeResults.anomaly_score * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-900 mb-2 text-sm">Combined Score</h4>
              <p className="text-lg font-bold text-danger-600">
                {(realTimeResults.combined_score * 100).toFixed(2)}%
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Flagged Transactions Table */}
      {batchResults?.flaggedTransactions && (
        <div className="card">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-4 sm:mb-6 space-y-2 sm:space-y-0">
            <h2 className="text-lg sm:text-xl font-semibold text-gray-900">Flagged Transactions</h2>
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 sm:h-5 sm:w-5 text-gray-400" />
              <span className="text-xs sm:text-sm text-gray-600">
                {batchResults.flaggedTransactions.length} transactions
              </span>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <div className="min-w-full">
              {/* Mobile Card View */}
              <div className="sm:hidden space-y-3">
                {batchResults.flaggedTransactions.map((tx, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-xs text-gray-500 uppercase tracking-wide">Transaction ID</p>
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {tx.Transaction_ID || tx.transaction_id}
                        </p>
                      </div>
                      <span className="status-badge status-fraud text-xs">Fraudulent</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3 text-xs mb-2">
                      <div>
                        <p className="text-gray-500">User ID</p>
                        <p className="font-medium truncate">{tx.User_ID || tx.user_id}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Amount</p>
                        <p className="font-medium">${tx.Transaction_Amount || tx.amount}</p>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div 
                            className="bg-red-500 h-2 rounded-full"
                            style={{ width: `${(tx.suspicion_score * 100)}%` }}
                          ></div>
                        </div>
                        <span className="text-xs">{(tx.suspicion_score * 100).toFixed(1)}%</span>
                      </div>
                      <button 
                        onClick={() => handleViewTransaction(tx)}
                        className="text-primary-600 hover:text-primary-900 p-1 rounded hover:bg-primary-50 transition-colors duration-200"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Desktop Table View */}
              <table className="hidden sm:table min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Transaction ID
                    </th>
                    <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      User ID
                    </th>
                    <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Amount
                    </th>
                    <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Suspicion Score
                    </th>
                    <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-3 sm:px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {batchResults.flaggedTransactions.map((tx, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-3 sm:px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        <span className="truncate block max-w-32">
                          {tx.Transaction_ID || tx.transaction_id}
                        </span>
                      </td>
                      <td className="px-3 sm:px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className="truncate block max-w-24">
                          {tx.User_ID || tx.user_id}
                        </span>
                      </td>
                      <td className="px-3 sm:px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        ${tx.Transaction_Amount || tx.amount}
                      </td>
                      <td className="px-3 sm:px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                            <div 
                              className="bg-red-500 h-2 rounded-full"
                              style={{ width: `${(tx.suspicion_score * 100)}%` }}
                            ></div>
                          </div>
                          <span>{(tx.suspicion_score * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                      <td className="px-3 sm:px-6 py-4 whitespace-nowrap">
                        <span className="status-badge status-fraud">
                          Fraudulent
                        </span>
                      </td>
                      <td className="px-3 sm:px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <button 
                          onClick={() => handleViewTransaction(tx)}
                          className="text-primary-600 hover:text-primary-900 p-1 rounded hover:bg-primary-50 transition-colors duration-200"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Transaction Details Modal */}
      {isModalOpen && selectedTransaction && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Transaction Details</h2>
              <button
                onClick={closeModal}
                className="text-gray-400 hover:text-gray-600 p-1 rounded-full hover:bg-gray-100 transition-colors duration-200"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            
            <div className="p-6 space-y-6">
              {/* Risk Assessment */}
              <div className={`p-4 rounded-lg ${getRiskLevel(selectedTransaction.suspicion_score).bgColor}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900">Risk Assessment</h3>
                    <p className={`text-lg font-bold ${getRiskLevel(selectedTransaction.suspicion_score).color}`}>
                      {getRiskLevel(selectedTransaction.suspicion_score).level}
                    </p>
                  </div>
                  <AlertTriangle className={`h-8 w-8 ${getRiskLevel(selectedTransaction.suspicion_score).color}`} />
                </div>
              </div>

              {/* Transaction Information */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-3">
                  <div className="flex items-center space-x-3">
                    <Hash className="h-5 w-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">Transaction ID</p>
                      <p className="font-medium text-gray-900">{selectedTransaction.Transaction_ID || selectedTransaction.transaction_id}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <User className="h-5 w-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">User ID</p>
                      <p className="font-medium text-gray-900">{selectedTransaction.User_ID || selectedTransaction.user_id}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <DollarSign className="h-5 w-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">Amount</p>
                      <p className="font-medium text-gray-900">${selectedTransaction.Transaction_Amount || selectedTransaction.amount}</p>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div>
                    <p className="text-sm text-gray-500 mb-2">Suspicion Score</p>
                    <div className="bg-gray-200 rounded-full h-3 mb-2">
                      <div 
                        className="bg-red-500 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${(selectedTransaction.suspicion_score * 100)}%` }}
                      ></div>
                    </div>
                    <p className="text-lg font-bold text-gray-900">
                      {(selectedTransaction.suspicion_score * 100).toFixed(2)}%
                    </p>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <Calendar className="h-5 w-5 text-gray-400" />
                    <div>
                      <p className="text-sm text-gray-500">Transaction Type</p>
                      <p className="font-medium text-gray-900">{selectedTransaction.Transaction_Type || 'N/A'}</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Additional Details */}
              {selectedTransaction.Merchant_Category && (
                <div className="border-t border-gray-200 pt-4">
                  <h4 className="font-semibold text-gray-900 mb-3">Additional Information</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Merchant Category</p>
                      <p className="font-medium text-gray-900">{selectedTransaction.Merchant_Category}</p>
                    </div>
                    {selectedTransaction.Location && (
                      <div>
                        <p className="text-gray-500">Location</p>
                        <p className="font-medium text-gray-900">{selectedTransaction.Location}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            <div className="flex justify-end p-6 border-t border-gray-200">
              <button
                onClick={closeModal}
                className="btn-secondary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Results; 