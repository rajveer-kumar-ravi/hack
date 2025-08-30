import React, { useState } from 'react';
import { Zap, AlertTriangle, CheckCircle, Loader, Shield } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const RealTimeAnalysis = () => {
  const { analyzeRealTimeTransaction, loading, error, realTimeResults } = useFraudDetection();
  const [formData, setFormData] = useState({
    Transaction_ID: '',
    User_ID: '',
    Transaction_Amount: '',
    Transaction_Type: 'CASH_IN',
    Merchant_Category: '',
    Location: '',
    Time_of_Day: '',
    Day_of_Week: '',
    Transaction_Frequency: '',
    Average_Transaction_Amount: '',
    Account_Age: '',
    Risk_Score: ''
  });

  const transactionTypes = [
    'CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT'
  ];

  const merchantCategories = [
    'FOOD_AND_DRINK', 'SHOPPING', 'TRANSPORT', 'ENTERTAINMENT', 
    'UTILITIES', 'HEALTHCARE', 'EDUCATION', 'TRAVEL', 'OTHER'
  ];

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await analyzeRealTimeTransaction(formData);
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
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Real-Time Analysis
        </h1>
        <p className="text-lg text-gray-600">
          Analyze individual transactions for fraud detection in real-time
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Transaction Details</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="Transaction_ID" className="block text-sm font-medium text-gray-700 mb-2">
                  Transaction ID *
                </label>
                <input
                  type="text"
                  id="Transaction_ID"
                  name="Transaction_ID"
                  value={formData.Transaction_ID}
                  onChange={handleInputChange}
                  required
                  className="input-field"
                  placeholder="Enter transaction ID"
                />
              </div>
              <div>
                <label htmlFor="User_ID" className="block text-sm font-medium text-gray-700 mb-2">
                  User ID *
                </label>
                <input
                  type="text"
                  id="User_ID"
                  name="User_ID"
                  value={formData.User_ID}
                  onChange={handleInputChange}
                  required
                  className="input-field"
                  placeholder="Enter user ID"
                />
              </div>
            </div>

            <div>
              <label htmlFor="Transaction_Amount" className="block text-sm font-medium text-gray-700 mb-2">
                Transaction Amount *
              </label>
              <input
                type="number"
                id="Transaction_Amount"
                name="Transaction_Amount"
                value={formData.Transaction_Amount}
                onChange={handleInputChange}
                required
                step="0.01"
                min="0"
                className="input-field"
                placeholder="Enter amount"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="Transaction_Type" className="block text-sm font-medium text-gray-700 mb-2">
                  Transaction Type
                </label>
                <select
                  id="Transaction_Type"
                  name="Transaction_Type"
                  value={formData.Transaction_Type}
                  onChange={handleInputChange}
                  className="input-field"
                >
                  {transactionTypes.map(type => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>
              <div>
                <label htmlFor="Merchant_Category" className="block text-sm font-medium text-gray-700 mb-2">
                  Merchant Category
                </label>
                <select
                  id="Merchant_Category"
                  name="Merchant_Category"
                  value={formData.Merchant_Category}
                  onChange={handleInputChange}
                  className="input-field"
                >
                  <option value="">Select category</option>
                  {merchantCategories.map(category => (
                    <option key={category} value={category}>{category}</option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <label htmlFor="Location" className="block text-sm font-medium text-gray-700 mb-2">
                Location
              </label>
              <input
                type="text"
                id="Location"
                name="Location"
                value={formData.Location}
                onChange={handleInputChange}
                className="input-field"
                placeholder="Enter location"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="Time_of_Day" className="block text-sm font-medium text-gray-700 mb-2">
                  Time of Day
                </label>
                <input
                  type="time"
                  id="Time_of_Day"
                  name="Time_of_Day"
                  value={formData.Time_of_Day}
                  onChange={handleInputChange}
                  className="input-field"
                />
              </div>
              <div>
                <label htmlFor="Day_of_Week" className="block text-sm font-medium text-gray-700 mb-2">
                  Day of Week
                </label>
                <select
                  id="Day_of_Week"
                  name="Day_of_Week"
                  value={formData.Day_of_Week}
                  onChange={handleInputChange}
                  className="input-field"
                >
                  <option value="">Select day</option>
                  <option value="Monday">Monday</option>
                  <option value="Tuesday">Tuesday</option>
                  <option value="Wednesday">Wednesday</option>
                  <option value="Thursday">Thursday</option>
                  <option value="Friday">Friday</option>
                  <option value="Saturday">Saturday</option>
                  <option value="Sunday">Sunday</option>
                </select>
              </div>
            </div>

            {/* Behavioral Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label htmlFor="Transaction_Frequency" className="block text-sm font-medium text-gray-700 mb-2">
                  Transaction Frequency
                </label>
                <input
                  type="number"
                  id="Transaction_Frequency"
                  name="Transaction_Frequency"
                  value={formData.Transaction_Frequency}
                  onChange={handleInputChange}
                  min="0"
                  className="input-field"
                  placeholder="Per day"
                />
              </div>
              <div>
                <label htmlFor="Average_Transaction_Amount" className="block text-sm font-medium text-gray-700 mb-2">
                  Avg Transaction Amount
                </label>
                <input
                  type="number"
                  id="Average_Transaction_Amount"
                  name="Average_Transaction_Amount"
                  value={formData.Average_Transaction_Amount}
                  onChange={handleInputChange}
                  step="0.01"
                  min="0"
                  className="input-field"
                  placeholder="Average amount"
                />
              </div>
              <div>
                <label htmlFor="Account_Age" className="block text-sm font-medium text-gray-700 mb-2">
                  Account Age (days)
                </label>
                <input
                  type="number"
                  id="Account_Age"
                  name="Account_Age"
                  value={formData.Account_Age}
                  onChange={handleInputChange}
                  min="0"
                  className="input-field"
                  placeholder="Days"
                />
              </div>
            </div>

            <div>
              <label htmlFor="Risk_Score" className="block text-sm font-medium text-gray-700 mb-2">
                Risk Score (0-100)
              </label>
              <input
                type="number"
                id="Risk_Score"
                name="Risk_Score"
                value={formData.Risk_Score}
                onChange={handleInputChange}
                min="0"
                max="100"
                className="input-field"
                placeholder="Risk score"
              />
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-danger-50 border border-danger-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <AlertTriangle className="h-5 w-5 text-danger-600" />
                  <p className="text-danger-800">{error}</p>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 px-4 rounded-lg font-medium transition-colors duration-200 ${
                loading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'btn-primary'
              }`}
            >
              {loading ? (
                <div className="flex items-center justify-center space-x-2">
                  <Loader className="h-5 w-5 animate-spin" />
                  <span>Analyzing...</span>
                </div>
              ) : (
                <div className="flex items-center justify-center space-x-2">
                  <Zap className="h-5 w-5" />
                  <span>Analyze Transaction</span>
                </div>
              )}
            </button>
          </form>
        </div>

        {/* Results Display */}
        <div className="space-y-6">
          {realTimeResults ? (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Analysis Results</h2>
              
              {/* Risk Assessment */}
              <div className="mb-6">
                <div className={`p-4 rounded-lg ${getRiskLevel(realTimeResults.fraud_probability).bgColor}`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-semibold text-gray-900">Risk Assessment</h3>
                      <p className={`text-lg font-bold ${getRiskLevel(realTimeResults.fraud_probability).color}`}>
                        {getRiskLevel(realTimeResults.fraud_probability).level}
                      </p>
                    </div>
                    <Shield className={`h-8 w-8 ${getRiskLevel(realTimeResults.fraud_probability).color}`} />
                  </div>
                </div>
              </div>

              {/* Fraud Probability */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-3">Fraud Probability</h3>
                <div className="bg-gray-200 rounded-full h-4 mb-2">
                  <div 
                    className="bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-4 rounded-full transition-all duration-500"
                    style={{ width: `${realTimeResults.fraud_probability * 100}%` }}
                  ></div>
                </div>
                <p className="text-2xl font-bold text-gray-900">
                  {(realTimeResults.fraud_probability * 100).toFixed(2)}%
                </p>
              </div>

              {/* Confidence Score */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-2">Model Confidence</h3>
                <p className="text-lg font-bold text-primary-600">
                  {(realTimeResults.confidence * 100).toFixed(2)}%
                </p>
              </div>

              {/* Recommendation */}
              <div className="mb-6">
                <h3 className="font-semibold text-gray-900 mb-2">Recommendation</h3>
                <p className="text-gray-700 bg-gray-50 p-3 rounded-lg">
                  {getRecommendation(realTimeResults.fraud_probability)}
                </p>
              </div>

              {/* Detailed Scores */}
              <div className="space-y-3">
                <h3 className="font-semibold text-gray-900">Detailed Analysis</h3>
                <div className="grid grid-cols-1 gap-3">
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-700">Supervised Score</span>
                    <span className="font-semibold text-gray-900">
                      {(realTimeResults.supervised_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-700">Anomaly Score</span>
                    <span className="font-semibold text-gray-900">
                      {(realTimeResults.anomaly_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-700">Combined Score</span>
                    <span className="font-semibold text-gray-900">
                      {(realTimeResults.combined_score * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card">
              <div className="text-center py-12">
                <Shield className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Ready for Analysis
                </h3>
                <p className="text-gray-600">
                  Fill in the transaction details and click "Analyze Transaction" to get started.
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