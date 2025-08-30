import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, CheckCircle, Loader } from 'lucide-react';
import { useFraudDetection } from '../context/FraudDetectionContext';

const BatchAnalysis = () => {
  const { runBatchAnalysis, loading, error, batchResults } = useFraudDetection();
  const [sampleSize, setSampleSize] = useState('');
  const [uploadedFile, setUploadedFile] = useState(null);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file && file.type === 'text/csv') {
      setUploadedFile(file);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    multiple: false
  });

  const handleAnalysis = async () => {
    if (!uploadedFile) return;
    
    try {
      const size = sampleSize ? parseInt(sampleSize) : null;
      await runBatchAnalysis(uploadedFile, size);
    } catch (error) {
      console.error('Analysis failed:', error);
    }
  };

  const getFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Batch Analysis
        </h1>
        <p className="text-lg text-gray-600">
          Upload a CSV file to perform bulk fraud detection analysis
        </p>
      </div>

      {/* File Upload */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Upload Dataset</h2>
        
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors duration-200 ${
            isDragActive
              ? 'border-primary-400 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          {isDragActive ? (
            <p className="text-primary-600 font-medium">Drop the CSV file here...</p>
          ) : (
            <div>
              <p className="text-gray-600 mb-2">
                Drag and drop a CSV file here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports CSV files with transaction data
              </p>
            </div>
          )}
        </div>

        {/* File Info */}
        {uploadedFile && (
          <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
            <div className="flex items-center space-x-3">
              <FileText className="h-5 w-5 text-green-600" />
              <div className="flex-1">
                <p className="font-medium text-green-900">{uploadedFile.name}</p>
                <p className="text-sm text-green-700">
                  Size: {getFileSize(uploadedFile.size)}
                </p>
              </div>
              <CheckCircle className="h-5 w-5 text-green-600" />
            </div>
          </div>
        )}

        {/* Sample Size Input */}
        <div className="mt-6">
          <label htmlFor="sampleSize" className="block text-sm font-medium text-gray-700 mb-2">
            Sample Size (Optional)
          </label>
          <input
            type="number"
            id="sampleSize"
            value={sampleSize}
            onChange={(e) => setSampleSize(e.target.value)}
            placeholder="Leave empty for full dataset"
            className="input-field"
          />
          <p className="text-sm text-gray-500 mt-1">
            Specify a number to analyze only a subset of the data for faster processing
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mt-6 p-4 bg-danger-50 border border-danger-200 rounded-lg">
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-5 w-5 text-danger-600" />
              <p className="text-danger-800">{error}</p>
            </div>
          </div>
        )}

        {/* Analysis Button */}
        <div className="mt-6">
          <button
            onClick={handleAnalysis}
            disabled={!uploadedFile || loading}
            className={`w-full py-3 px-4 rounded-lg font-medium transition-colors duration-200 ${
              !uploadedFile || loading
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
              'Start Analysis'
            )}
          </button>
        </div>
      </div>

      {/* Results Preview */}
      {batchResults && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Analysis Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <p className="text-2xl font-bold text-blue-600">
                {batchResults.statistics?.totalTransactions || 0}
              </p>
              <p className="text-sm text-gray-600">Total Transactions</p>
            </div>
            <div className="text-center p-4 bg-danger-50 rounded-lg">
              <p className="text-2xl font-bold text-danger-600">
                {batchResults.statistics?.fraudulentTransactions || 0}
              </p>
              <p className="text-sm text-gray-600">Fraudulent</p>
            </div>
            <div className="text-center p-4 bg-success-50 rounded-lg">
              <p className="text-2xl font-bold text-success-600">
                {batchResults.statistics?.legitimateTransactions || 0}
              </p>
              <p className="text-sm text-gray-600">Legitimate</p>
            </div>
            <div className="text-center p-4 bg-warning-50 rounded-lg">
              <p className="text-2xl font-bold text-warning-600">
                {((batchResults.statistics?.accuracy || 0) * 100).toFixed(2)}%
              </p>
              <p className="text-sm text-gray-600">Accuracy</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Precision</h3>
              <p className="text-2xl font-bold text-primary-600">
                {((batchResults.statistics?.precision || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">Recall</h3>
              <p className="text-2xl font-bold text-primary-600">
                {((batchResults.statistics?.recall || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <h3 className="font-semibold text-gray-900 mb-2">F1 Score</h3>
              <p className="text-2xl font-bold text-primary-600">
                {((batchResults.statistics?.f1Score || 0) * 100).toFixed(2)}%
              </p>
            </div>
          </div>

          {batchResults.flaggedTransactions && (
            <div className="mt-6">
              <h3 className="font-semibold text-gray-900 mb-4">Flagged Transactions</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Transaction ID
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
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {batchResults.flaggedTransactions.slice(0, 10).map((tx, index) => (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {tx.Transaction_ID || tx.transaction_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          ${tx.Transaction_Amount || tx.amount}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {(tx.suspicion_score * 100).toFixed(2)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="status-badge status-fraud">
                            Fraudulent
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {batchResults.flaggedTransactions.length > 10 && (
                <p className="text-sm text-gray-500 mt-2">
                  Showing first 10 of {batchResults.flaggedTransactions.length} flagged transactions
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default BatchAnalysis; 