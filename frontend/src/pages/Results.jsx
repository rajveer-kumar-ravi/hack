import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, PieChart, Pie, Cell } from 'recharts';
import { Download, Eye, Filter, TrendingUp, AlertTriangle, CheckCircle, X, Calendar, DollarSign, User, Hash } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const Results = () => {
  const { batchResults, realTimeResults } = useFraudDetection();
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const prepareChartData = () => {
    if (!batchResults?.statistics) return [];
    
    const legitimate = batchResults.statistics.legitimateTransactions;
    const fraudulent = batchResults.statistics.fraudulentTransactions;
    const total = legitimate + fraudulent;
    
    const data = [
      { 
        name: 'Legitimate', 
        value: legitimate, 
        color: '#10B981',
        percentage: ((legitimate / total) * 100).toFixed(1)
      },
      { 
        name: 'Fraudulent', 
        value: fraudulent, 
        color: '#EF4444',
        percentage: ((fraudulent / total) * 100).toFixed(1)
      }
    ];
    
    return data;
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
         `${tx.Transaction_ID || tx.transaction_id || 'N/A'},${tx.User_ID || tx.user_id || 'N/A'},${formatAmount(tx)},${tx.suspicion_score || 0},Fraudulent`
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

  const formatAmount = (transaction) => {
    // Based on your dataset structure, the amount is likely in one of these fields
    // Let's check the most common ones first
    const amount = transaction.Amount || 
                   transaction.amount || 
                   transaction.Transaction_Amount ||
                   transaction.transaction_amount ||
                   transaction.TransactionAmount ||
                   transaction.transactionAmount ||
                   // If none of the above work, try V fields (common in credit card fraud datasets)
                   transaction.V1 ||
                   transaction.V2 ||
                   transaction.V3 ||
                   transaction.V4 ||
                   transaction.V5 ||
                   transaction.V6 ||
                   transaction.V7 ||
                   transaction.V8 ||
                   transaction.V9 ||
                   transaction.V10 ||
                   transaction.V11 ||
                   transaction.V12 ||
                   transaction.V13 ||
                   transaction.V14 ||
                   transaction.V15 ||
                   transaction.V16 ||
                   transaction.V17 ||
                   transaction.V18 ||
                   transaction.V19 ||
                   transaction.V20 ||
                   transaction.V21 ||
                   transaction.V22 ||
                   transaction.V23 ||
                   transaction.V24 ||
                   transaction.V25 ||
                   transaction.V26 ||
                   transaction.V27 ||
                   transaction.V28 ||
                   0;
    // If amount is 0 or falsy, return a placeholder
    if (!amount || amount === 0) {
      return '0.00';
    }
    
    // Format the amount properly
    return typeof amount === 'number' ? amount.toFixed(2) : amount;
  };

  

  // if (!batchResults && !realTimeResults) {
  //   return (
  //     <div className="max-w-4xl mx-auto text-center py-8 sm:py-12">
  //       <div className="card">
  //         <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-3 sm:mb-4">No Results Available</h1>
  //         <p className="text-base sm:text-lg text-gray-600 mb-6 sm:mb-8 px-4">
  //           Run a batch analysis or real-time analysis to see results here.
  //         </p>
  //         <div className="flex flex-col sm:flex-row justify-center space-y-3 sm:space-y-0 sm:space-x-4">
  //           <button className="btn-primary">Run Batch Analysis</button>
  //           <button className="btn-secondary">Real-Time Analysis</button>
  //         </div>
  //       </div>
  //     </div>
  //   );
  // }

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
             <div className="flex justify-center p-4">
               <PieChart width={Math.min(500, window.innerWidth - 80)} height={400}> {/* Increased width and height */}
                 <Pie
                   data={prepareChartData()}
                   cx="50%"
                   cy="50%"
                   labelLine={true}
                                       label={({ name, value, cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
                      const RADIAN = Math.PI / 180;
                      const radius = outerRadius + 15;
                      const x = cx + radius * Math.cos(-midAngle * RADIAN);
                      const y = cy + radius * Math.sin(-midAngle * RADIAN);
                      
                      return (
                        <text 
                          x={x} 
                          y={y} 
                          fill="#374151" 
                          textAnchor={x > cx ? 'start' : 'end'} 
                          dominantBaseline="central"
                          style={{ fontSize: '12px', fontWeight: 'bold' }}
                        >
                          {`${name}\n${(percent * 100).toFixed(1)}%`}
                        </text>
                      );
                    }}
                   outerRadius={110}
                   fill="#8884d8"
                   dataKey="value"
                   minAngle={3}
                 >
                   {prepareChartData().map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={entry.color} />
                   ))}
                 </Pie>
                                   <Tooltip 
                    formatter={(value, name, props) => {
                      const total = batchResults.statistics.legitimateTransactions + batchResults.statistics.fraudulentTransactions;
                      const percentage = ((value / total) * 100).toFixed(1);
                      return [`${name}: ${value} (${percentage}%)`, 'Count'];
                    }}
                  />
               </PieChart>
             </div>
           </div>

                     {/* Model Performance Metrics */}
           <div className="card">
             <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Model Performance</h2>
             <div className="flex justify-center p-4">
               <BarChart width={Math.min(500, window.innerWidth - 80)} height={400} data={prepareMetricsData()}>
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
                 {realTimeResults.anomaly_score !== null && realTimeResults.anomaly_score !== undefined 
                   ? (realTimeResults.anomaly_score * 100).toFixed(2) + '%'
                   : 'N/A'}
               </p>
             </div>
             <div className="p-3 bg-gray-50 rounded-lg">
               <h4 className="font-semibold text-gray-900 mb-2 text-sm">Combined Score</h4>
               <p className="text-lg font-bold text-danger-600">
                 {realTimeResults.combined_score !== null && realTimeResults.combined_score !== undefined 
                   ? (realTimeResults.combined_score * 100).toFixed(2) + '%'
                   : 'N/A'}
               </p>
             </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default Results; 