import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';

const FraudDetectionContext = createContext();

const initialState = {
  loading: false,
  error: null,
  results: null,
  batchResults: null,
  realTimeResults: null,
  modelStatus: 'idle',
  statistics: {
    totalTransactions: 0,
    fraudulentTransactions: 0,
    legitimateTransactions: 0,
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1Score: 0
  }
};

const fraudDetectionReducer = (state, action) => {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'SET_RESULTS':
      return { 
        ...state, 
        results: action.payload, 
        loading: false, 
        error: null 
      };
    case 'SET_BATCH_RESULTS':
      return { 
        ...state, 
        batchResults: action.payload, 
        loading: false, 
        error: null 
      };
    case 'SET_REALTIME_RESULTS':
      return { 
        ...state, 
        realTimeResults: action.payload, 
        loading: false, 
        error: null 
      };
    case 'SET_MODEL_STATUS':
      return { ...state, modelStatus: action.payload };
    case 'SET_STATISTICS':
      return { ...state, statistics: action.payload };
    case 'CLEAR_RESULTS':
      return { 
        ...state, 
        results: null, 
        batchResults: null, 
        realTimeResults: null,
        error: null 
      };
    default:
      return state;
  }
};

export const FraudDetectionProvider = ({ children }) => {
  const [state, dispatch] = useReducer(fraudDetectionReducer, initialState);

  const api = axios.create({
    baseURL: 'http://localhost:5000',
    timeout: 300000, // 5 minutes for batch processing
  });

  const runBatchAnalysis = async (file, sampleSize = null) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (sampleSize) {
        formData.append('sample_size', sampleSize);
      }

      const response = await api.post('/api/batch-analysis', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      dispatch({ type: 'SET_BATCH_RESULTS', payload: response.data });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'An error occurred during batch analysis';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    }
  };

  const analyzeRealTimeTransaction = async (transactionData) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'SET_ERROR', payload: null });
    
    try {
      const response = await api.post('/api/real-time-analysis', transactionData);
      dispatch({ type: 'SET_REALTIME_RESULTS', payload: response.data });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'An error occurred during real-time analysis';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    }
  };

  const getModelStatus = async () => {
    try {
      const response = await api.get('/api/model-status');
      dispatch({ type: 'SET_MODEL_STATUS', payload: response.data.status });
      return response.data;
    } catch (error) {
      console.error('Error fetching model status:', error);
    }
  };

  const clearResults = () => {
    dispatch({ type: 'CLEAR_RESULTS' });
  };

  useEffect(() => {
    getModelStatus();
  }, []);

  const value = {
    ...state,
    runBatchAnalysis,
    analyzeRealTimeTransaction,
    getModelStatus,
    clearResults,
  };

  return (
    <FraudDetectionContext.Provider value={value}>
      {children}
    </FraudDetectionContext.Provider>
  );
};

export const useFraudDetection = () => {
  const context = useContext(FraudDetectionContext);
  if (!context) {
    throw new Error('useFraudDetection must be used within a FraudDetectionProvider');
  }
  return context;
}; 