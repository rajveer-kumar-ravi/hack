import React, { useState, useEffect } from 'react';
import { Zap, AlertTriangle, Loader, Shield, Play, Square, Activity, BarChart3 } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const RealTimeAnalysis = () => {
  const { analyzeRealTimeTransaction, loading, error, realTimeResults, modelStatus } = useFraudDetection();
  
  // Simulation state
  const [simulationStatus, setSimulationStatus] = useState({
    running: false,
    totalTransactions: 0,
    latestTransactions: [],
    error: null
  });
  const [simulationResults, setSimulationResults] = useState(null);
  const [simulationLoading, setSimulationLoading] = useState(false);
  
  const [formData, setFormData] = useState({
    Amount: '',
    V1: '',
    V2: '',
    V3: '',
    V4: '',
    V5: '',
    V6: '',
    V7: '',
    V8: '',
    V9: '',
    V10: '',
    V11: '',
    V12: '',
    V13: '',
    V14: '',
    V15: '',
    V16: '',
    V17: '',
    V18: '',
    V19: '',
    V20: '',
    V21: '',
    V22: '',
    V23: '',
    V24: '',
    V25: '',
    V26: '',
    V27: '',
    V28: ''
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Simulation functions
  const fetchSimulationStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/simulation/status');
      const data = await response.json();
      console.log('Simulation status received:', data); // Debug log
      setSimulationStatus(data);
    } catch (error) {
      console.error('Error fetching simulation status:', error);
      setSimulationStatus(prev => ({ ...prev, error: error.message }));
    }
  };

  const startSimulation = async () => {
    setSimulationLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/simulation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error('Failed to start simulation');
      }
      
      await fetchSimulationStatus();
      setSimulationResults(null); // Clear previous results
    } catch (error) {
      console.error('Error starting simulation:', error);
      setSimulationStatus(prev => ({ ...prev, error: error.message }));
    } finally {
      setSimulationLoading(false);
    }
  };

  const stopSimulation = async () => {
    setSimulationLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/simulation/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error('Failed to stop simulation');
      }
      
      await fetchSimulationStatus();
    } catch (error) {
      console.error('Error stopping simulation:', error);
      setSimulationStatus(prev => ({ ...prev, error: error.message }));
    } finally {
      setSimulationLoading(false);
    }
  };

  const analyzeSimulationData = async () => {
    setSimulationLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/simulation/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to analyze simulation data');
      }
      
      const data = await response.json();
      setSimulationResults(data);
    } catch (error) {
      console.error('Error analyzing simulation data:', error);
      alert(`Analysis failed: ${error.message}`);
    } finally {
      setSimulationLoading(false);
    }
  };

  // Poll simulation status when running
  useEffect(() => {
    let interval;
    if (simulationStatus.running) {
      interval = setInterval(fetchSimulationStatus, 1000); // Poll every 1 second when running
    } else {
      // Also poll occasionally when stopped to get updated counts
      interval = setInterval(fetchSimulationStatus, 5000); // Poll every 5 seconds when stopped
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [simulationStatus.running]);

  // Initial status fetch
  useEffect(() => {
    fetchSimulationStatus();
  }, []);

  // Check if all required fields are filled
  const isFormValid = () => {
    // Amount is required and must be a valid number > 0
    const amountValid = formData.Amount && !isNaN(parseFloat(formData.Amount)) && parseFloat(formData.Amount) > 0;
    
    // All V fields must have values (cannot be empty strings)
    const vFieldsValid = Array.from({ length: 28 }, (_, i) => i + 1).every(num => {
      const value = formData[`V${num}`];
      return value !== '' && value !== null && value !== undefined;
    });
    
    return amountValid && vFieldsValid;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Convert numeric fields
      const processedData = {
        ...formData,
        Amount: parseFloat(formData.Amount) || 0,
        V1: parseFloat(formData.V1) || 0,
        V2: parseFloat(formData.V2) || 0,
        V3: parseFloat(formData.V3) || 0,
        V4: parseFloat(formData.V4) || 0,
        V5: parseFloat(formData.V5) || 0,
        V6: parseFloat(formData.V6) || 0,
        V7: parseFloat(formData.V7) || 0,
        V8: parseFloat(formData.V8) || 0,
        V9: parseFloat(formData.V9) || 0,
        V10: parseFloat(formData.V10) || 0,
        V11: parseFloat(formData.V11) || 0,
        V12: parseFloat(formData.V12) || 0,
        V13: parseFloat(formData.V13) || 0,
        V14: parseFloat(formData.V14) || 0,
        V15: parseFloat(formData.V15) || 0,
        V16: parseFloat(formData.V16) || 0,
        V17: parseFloat(formData.V17) || 0,
        V18: parseFloat(formData.V18) || 0,
        V19: parseFloat(formData.V19) || 0,
        V20: parseFloat(formData.V20) || 0,
        V21: parseFloat(formData.V21) || 0,
        V22: parseFloat(formData.V22) || 0,
        V23: parseFloat(formData.V23) || 0,
        V24: parseFloat(formData.V24) || 0,
        V25: parseFloat(formData.V25) || 0,
        V26: parseFloat(formData.V26) || 0,
        V27: parseFloat(formData.V27) || 0,
        V28: parseFloat(formData.V28) || 0
      };
      
      await analyzeRealTimeTransaction(processedData);
    } catch (error) {
      console.error('Real-time analysis failed:', error);
    }
  };

  const getRiskLevel = (score) => {
    if (score >= 0.8) return { level: 'High Risk', color: 'text-danger-600', bgColor: 'bg-danger-50' };
    if (score >= 0.5) return { level: 'Medium Risk', color: 'text-warning-600', bgColor: 'bg-warning-50' };
    return { level: 'Low Risk', color: 'text-success-600', bgColor: 'bg-success-50' };
  };

  const getRecommendation = (score) => {
    if (score >= 0.8) return 'Immediate action required. Block transaction and flag account.';
    if (score >= 0.5) return 'Additional verification recommended. Review transaction details.';
    return 'Transaction appears legitimate. Standard processing recommended.';
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 sm:space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-3 sm:mb-4">
          Real-Time Analysis
        </h1>
        <p className="text-base sm:text-lg text-gray-600 px-4">
          Analyze individual credit card transactions or run real-time simulation
        </p>
      </div>

      {/* Real-Time Simulation Section */}
      {/* <div className="card">
        <div className="flex items-center space-x-3 mb-6">
          <Activity className="h-6 w-6 text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Real-Time Transaction Simulation</h2>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div>
                <p className="text-sm font-medium text-gray-700">Simulation Status</p>
                <p className={`text-lg font-semibold ${simulationStatus.running ? 'text-green-600' : 'text-gray-500'}`}>
                  {simulationStatus.running ? 'Running' : 'Stopped'}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm font-medium text-gray-700">Total Transactions</p>
                <p className="text-lg font-semibold text-blue-600">{simulationStatus.totalTransactions || 0}</p>
              </div>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={startSimulation}
                disabled={simulationStatus.running || simulationLoading}
                className={`flex-1 btn-primary flex items-center justify-center space-x-2 ${
                  simulationStatus.running || simulationLoading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                <Play className="h-4 w-4" />
                <span>{simulationLoading && !simulationStatus.running ? 'Starting...' : 'Start Simulation'}</span>
              </button>
              
              <button
                onClick={stopSimulation}
                disabled={!simulationStatus.running || simulationLoading}
                className={`flex-1 btn-secondary flex items-center justify-center space-x-2 ${
                  !simulationStatus.running || simulationLoading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                <Square className="h-4 w-4" />
                <span>{simulationLoading && simulationStatus.running ? 'Stopping...' : 'Stop Simulation'}</span>
              </button>
            </div>

                         <button
               onClick={analyzeSimulationData}
               disabled={simulationStatus.totalTransactions === 0 || simulationLoading || modelStatus?.status !== 'ready'}
               className={`w-full btn-primary flex items-center justify-center space-x-2 ${
                 simulationStatus.totalTransactions === 0 || simulationLoading || modelStatus?.status !== 'ready' 
                   ? 'opacity-50 cursor-not-allowed' : ''
               }`}
             >
               <BarChart3 className="h-4 w-4" />
               <span>{simulationLoading ? 'Analyzing...' : 'Start Analysis'}</span>
             </button>
             
             <button
               onClick={fetchSimulationStatus}
               className="w-full btn-secondary flex items-center justify-center space-x-2"
             >
               <Activity className="h-4 w-4" />
               <span>Refresh Status</span>
             </button>
            
            {(!modelStatus || modelStatus.status !== 'ready') && (
              <p className="text-sm text-amber-600 bg-amber-50 p-3 rounded-lg">
                ⚠️ Model not ready. Please run batch analysis first to train the model.
              </p>
            )}
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">Latest Transactions</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {simulationStatus.latestTransactions && simulationStatus.latestTransactions.length > 0 ? (
                simulationStatus.latestTransactions.slice().reverse().map((tx, index) => (
                  <div key={index} className="p-3 bg-gray-50 rounded-lg text-sm">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">${tx.Amount?.toFixed(2)}</span>
                      <span className="text-gray-500 text-xs">
                        {new Date(tx.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-xs text-gray-600 mt-1">
                      ID: {tx.transaction_id?.substring(0, 8)}...
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-sm text-center py-8">
                  No transactions generated yet. Start simulation to see live data.
                </p>
              )}
            </div>
          </div>
        </div>

        {simulationResults && (
          <div className="mt-6 p-6 bg-blue-50 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Simulation Analysis Results</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{simulationResults.statistics?.totalTransactions || 0}</p>
                <p className="text-sm text-gray-600">Total Transactions</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-red-600">{simulationResults.statistics?.flaggedTransactions || 0}</p>
                <p className="text-sm text-gray-600">Flagged as Fraud</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-yellow-600">{simulationResults.statistics?.fraudRate?.toFixed(1) || '0.0'}%</p>
                <p className="text-sm text-gray-600">Fraud Rate</p>
              </div>
            </div>

            {simulationResults.flaggedTransactions && simulationResults.flaggedTransactions.length > 0 && (
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">Flagged Transactions (Top 5)</h4>
                <div className="space-y-2 max-h-40 overflow-y-auto">
                  {simulationResults.flaggedTransactions.slice(0, 5).map((tx, index) => (
                    <div key={index} className="p-2 bg-white rounded border border-red-200">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">${tx.Amount?.toFixed(2) || '0.00'}</span>
                        <span className="text-red-600 font-medium">
                          {((tx.fraud_probability || 0) * 100).toFixed(1)}% fraud risk
                        </span>
                      </div>
                      <div className="text-xs text-gray-600">
                        ID: {tx.transaction_id?.substring(0, 12) || 'N/A'}...
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

                 {simulationStatus.error && (
           <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
             <p className="text-red-600 text-sm">{simulationStatus.error}</p>
           </div>
         )}
         
         <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
           <h4 className="text-sm font-medium text-gray-700 mb-2">Debug Info</h4>
           <div className="text-xs text-gray-600 space-y-1">
             <p>Running: {simulationStatus.running ? 'Yes' : 'No'}</p>
             <p>Total Transactions: {simulationStatus.totalTransactions || 0}</p>
             <p>Latest Transactions Count: {simulationStatus.latestTransactions ? simulationStatus.latestTransactions.length : 0}</p>
             <p>File Exists: {simulationStatus.file_exists ? 'Yes' : 'No'}</p>
             <p>Timestamp: {simulationStatus.timestamp || 'N/A'}</p>
           </div>
         </div>
      </div> */}

      {/* Model Status Warning */}
      {modelStatus === 'idle' && (
        <div className="card bg-warning-50 border-warning-200">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="h-5 w-5 text-warning-600 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-warning-900">Model Not Ready</h3>
              <p className="text-warning-800 text-sm">
                Please run batch analysis first to train the model before using real-time analysis.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
        {/* Input Form */}
        <div className="card">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Transaction Details</h2>
          
          <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
            {/* Amount */}
            <div>
              <label htmlFor="Amount" className="block text-sm font-medium text-gray-700 mb-2">
                Transaction Amount *
              </label>
                             <input
                 type="number"
                 id="Amount"
                 name="Amount"
                 value={formData.Amount}
                 onChange={handleInputChange}
                 required
                 step="0.01"
                 min="0"
                 className={`input-field ${
                   formData.Amount && !isNaN(parseFloat(formData.Amount)) && parseFloat(formData.Amount) > 0
                     ? 'border-green-500 focus:border-green-500'
                     : formData.Amount !== ''
                     ? 'border-red-500 focus:border-red-500'
                     : ''
                 }`}
                 placeholder="Enter amount"
               />
               {formData.Amount !== '' && (!formData.Amount || isNaN(parseFloat(formData.Amount)) || parseFloat(formData.Amount) <= 0) && (
                 <p className="text-red-500 text-xs mt-1">Please enter a valid amount greater than 0</p>
               )}
            </div>

                         {/* V1-V28 Features */}
             <div>
               <label className="block text-sm font-medium text-gray-700 mb-2">
                 Feature Values (V1-V28) *
               </label>
               <p className="text-xs text-gray-500 mb-3">
                 These are PCA-transformed features. <strong>All fields are required.</strong> You can use any numeric values including 0, but all fields must be filled.
               </p>
              <div className="grid grid-cols-4 gap-2">
                {Array.from({ length: 28 }, (_, i) => i + 1).map(num => (
                  <div key={num}>
                    <label htmlFor={`V${num}`} className="block text-xs text-gray-500 mb-1">
                      V{num}
                    </label>
                                         <input
                       type="number"
                       id={`V${num}`}
                       name={`V${num}`}
                       value={formData[`V${num}`]}
                       onChange={handleInputChange}
                       step="0.01"
                       className={`input-field text-xs py-1 ${
                         formData[`V${num}`] !== '' 
                           ? 'border-green-500 focus:border-green-500' 
                           : 'border-red-500 focus:border-red-500'
                       }`}
                       placeholder="-"
                       required
                     />
                  </div>
                ))}
              </div>
            </div>

            {/* Form Validation Status */}
            <div className={`p-3 sm:p-4 rounded-lg border ${
              isFormValid() 
                ? 'bg-green-50 border-green-200' 
                : 'bg-yellow-50 border-yellow-200'
            }`}>
              <div className="flex items-center space-x-3">
                {isFormValid() ? (
                  <Shield className="h-4 w-4 sm:h-5 sm:w-5 text-green-600 flex-shrink-0" />
                ) : (
                  <AlertTriangle className="h-4 w-4 sm:h-5 sm:w-5 text-yellow-600 flex-shrink-0" />
                )}
                <div>
                  <p className={`text-sm font-medium ${
                    isFormValid() ? 'text-green-900' : 'text-yellow-900'
                  }`}>
                    {isFormValid() ? 'All fields are valid!' : 'Please fill in all required fields'}
                  </p>
                                     <p className={`text-xs ${
                     isFormValid() ? 'text-green-700' : 'text-yellow-700'
                   }`}>
                     {isFormValid() 
                       ? 'You can now analyze the transaction' 
                       : `Transaction Amount is required and all V1-V28 fields must have values. Currently ${Array.from({ length: 28 }, (_, i) => i + 1).filter(num => !formData[`V${num}`] || formData[`V${num}`] === '').length} V fields are empty. Please fill in all fields to enable the Analyze Transaction button.`
                     }
                   </p>
                </div>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-3 sm:p-4 bg-danger-50 border border-danger-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <AlertTriangle className="h-4 w-4 sm:h-5 sm:w-5 text-danger-600 flex-shrink-0" />
                  <p className="text-danger-800 text-sm sm:text-base">{error}</p>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || modelStatus === 'idle' || !isFormValid()}
              className={`w-full py-3 px-4 rounded-lg font-medium transition-colors duration-200 ${
                loading || modelStatus === 'idle' || !isFormValid()
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'btn-primary'
              }`}
            >
              {loading ? (
                <div className="flex items-center justify-center space-x-2">
                  <Loader className="h-4 w-4 sm:h-5 sm:w-5 animate-spin" />
                  <span className="text-sm sm:text-base">Analyzing...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Zap className="h-4 w-4 sm:h-5 sm:w-5" />
                  <span className="text-sm sm:text-base">Analyze Transaction</span>
                </div>
              )}
            </button>
          </form>
        </div>

        {/* Results Display */}
        <div className="space-y-4 sm:space-y-6">
          {realTimeResults ? (
            <div className="card">
              <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-4 sm:mb-6">Analysis Results</h2>
              
              {/* Risk Assessment */}
              <div className="mb-4 sm:mb-6">
                <div className={`p-3 sm:p-4 rounded-lg ${getRiskLevel(realTimeResults.fraud_probability).bgColor}`}>
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-gray-900 text-sm sm:text-base">Risk Assessment</h3>
                      <p className={`text-base sm:text-lg font-bold ${getRiskLevel(realTimeResults.fraud_probability).color}`}>
                        {getRiskLevel(realTimeResults.fraud_probability).level}
                      </p>
                    </div>
                    <Shield className={`h-6 w-6 sm:h-8 sm:w-8 ${getRiskLevel(realTimeResults.fraud_probability).color} flex-shrink-0`} />
                  </div>
                </div>
              </div>

              {/* Fraud Probability */}
              <div className="mb-4 sm:mb-6">
                <h3 className="font-semibold text-gray-900 mb-2 sm:mb-3 text-sm sm:text-base">Fraud Probability</h3>
                <div className="bg-gray-200 rounded-full h-3 sm:h-4 mb-2">
                  <div 
                    className="bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-3 sm:h-4 rounded-full transition-all duration-500"
                    style={{ width: `${realTimeResults.fraud_probability * 100}%` }}
                  ></div>
                </div>
                <p className="text-xl sm:text-2xl font-bold text-gray-900">
                  {(realTimeResults.fraud_probability * 100).toFixed(2)}%
                </p>
              </div>

              {/* Confidence Score */}
              <div className="mb-4 sm:mb-6">
                <h3 className="font-semibold text-gray-900 mb-2 text-sm sm:text-base">Model Confidence</h3>
                <p className="text-lg sm:text-xl font-bold text-primary-600">
                  {(realTimeResults.confidence * 100).toFixed(2)}%
                </p>
              </div>

            </div>
          ) : (
            <div className="card">
              <div className="text-center py-8 sm:py-12">
                <Shield className="h-12 w-12 sm:h-16 sm:w-16 text-gray-400 mx-auto mb-3 sm:mb-4" />
                <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
                  {modelStatus === 'idle' ? 'Model Not Ready' : 'Ready for Analysis'}
                </h3>
                <p className="text-gray-600 text-sm sm:text-base px-4">
                  {modelStatus === 'idle' 
                    ? 'Please run batch analysis first to train the model.'
                    : 'Fill in the transaction details and click "Analyze Transaction" to get started.'
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RealTimeAnalysis; 