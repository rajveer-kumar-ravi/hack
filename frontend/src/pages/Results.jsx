import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { Download, Eye, Filter, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const Results = () => {
  const { batchResults, realTimeResults } = useFraudDetection();

  const COLORS = ['#10B981', '#F59E0B', '#EF4444'];

  const prepareChartData = () => {
    if (!batchResults?.statistics) return [];
    
    return [
      { name: 'Legitimate', value: batchResults.statistics.legitimateTransactions, color: '#10B981' },
      { name: 'Suspicious', value: batchResults.statistics.suspiciousTransactions || 0, color: '#F59E0B' },
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
    if (!batchResults) return;
    
    const csvContent = "data:text/csv;charset=utf-8," + 
      "Transaction_ID,User_ID,Transaction_Amount,Suspicion_Score,Status\n" +
      batchResults.flaggedTransactions?.map(tx => 
        `${tx.Transaction_ID || tx.transaction_id},${tx.User_ID || tx.user_id},${tx.Transaction_Amount || tx.amount},${tx.suspicion_score},Fraudulent`
      ).join('\n');
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "fraud_detection_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (!batchResults && !realTimeResults) {
    return (
      <div className="max-w-4xl mx-auto text-center py-12">
        <div className="card">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">No Results Available</h1>
          <p className="text-lg text-gray-600 mb-8">
            Run a batch analysis or real-time analysis to see results here.
          </p>
          <div className="flex justify-center space-x-4">
            <button className="btn-primary">Run Batch Analysis</button>
            <button className="btn-secondary">Real-Time Analysis</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Analysis Results</h1>
          <p className="text-lg text-gray-600">
            Detailed insights from fraud detection analysis
          </p>
        </div>
        {batchResults && (
          <button
            onClick={exportResults}
            className="btn-primary flex items-center space-x-2"
          >
            <Download className="h-5 w-5" />
            <span>Export Results</span>
          </button>
        )}
      </div>

      {/* Summary Cards */}
      {batchResults && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Transactions</p>
                <p className="text-2xl font-bold text-gray-900">
                  {batchResults.statistics?.totalTransactions || 0}
                </p>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg">
                <TrendingUp className="h-6 w-6 text-blue-600" />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Fraudulent</p>
                <p className="text-2xl font-bold text-danger-600">
                  {batchResults.statistics?.fraudulentTransactions || 0}
                </p>
              </div>
              <div className="p-3 bg-danger-50 rounded-lg">
                <AlertTriangle className="h-6 w-6 text-danger-600" />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Legitimate</p>
                <p className="text-2xl font-bold text-success-600">
                  {batchResults.statistics?.legitimateTransactions || 0}
                </p>
              </div>
              <div className="p-3 bg-success-50 rounded-lg">
                <CheckCircle className="h-6 w-6 text-success-600" />
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold text-warning-600">
                  {((batchResults.statistics?.accuracy || 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className="p-3 bg-warning-50 rounded-lg">
                <TrendingUp className="h-6 w-6 text-warning-600" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Charts */}
      {batchResults && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Transaction Distribution */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Transaction Distribution</h2>
            <PieChart width={400} height={300}>
              <Pie
                data={prepareChartData()}
                cx={200}
                cy={150}
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

          {/* Model Performance Metrics */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Model Performance</h2>
            <BarChart width={400} height={300} data={prepareMetricsData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#3B82F6" />
            </BarChart>
          </div>
        </div>
      )}

      {/* Real-Time Results */}
      {realTimeResults && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Real-Time Analysis Result</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Fraud Probability</h3>
              <p className="text-3xl font-bold text-danger-600">
                {(realTimeResults.fraud_probability * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Confidence</h3>
              <p className="text-3xl font-bold text-primary-600">
                {(realTimeResults.confidence * 100).toFixed(2)}%
              </p>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Risk Level</h3>
              <p className="text-2xl font-bold text-warning-600">
                {realTimeResults.fraud_probability >= 0.8 ? 'High' : 
                 realTimeResults.fraud_probability >= 0.5 ? 'Medium' : 'Low'}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Flagged Transactions Table */}
      {batchResults?.flaggedTransactions && (
        <div className="card">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Flagged Transactions</h2>
            <div className="flex items-center space-x-2">
              <Filter className="h-5 w-5 text-gray-400" />
              <span className="text-sm text-gray-600">
                {batchResults.flaggedTransactions.length} transactions
              </span>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Transaction ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    User ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Amount
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Suspicion Score
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {batchResults.flaggedTransactions.map((tx, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {tx.Transaction_ID || tx.transaction_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {tx.User_ID || tx.user_id}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      ${tx.Transaction_Amount || tx.amount}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
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
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="status-badge status-fraud">
                        Fraudulent
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <button className="text-primary-600 hover:text-primary-900">
                        <Eye className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default Results; 