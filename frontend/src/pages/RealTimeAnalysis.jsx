import React, { useState } from 'react';
import { Zap, AlertTriangle, Loader, Shield } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const RealTimeAnalysis = () => {
  const { analyzeRealTimeTransaction, loading, error, realTimeResults, modelStatus } = useFraudDetection();
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
          Analyze individual credit card transactions for fraud detection in real-time
        </p>
      </div>

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
                       placeholder="Enter value"
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