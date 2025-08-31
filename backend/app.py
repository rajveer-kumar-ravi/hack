from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import tempfile
import json
from werkzeug.utils import secure_filename
import pickle
from datetime import datetime
import time

# Import the fraud detection functions from the notebook code
import fraud_detection

app = Flask(__name__)
# Allow large CSV uploads. Default 2048MB (2GB). Override with env MAX_CONTENT_MB.
MAX_CONTENT_MB = int(os.environ.get('MAX_CONTENT_MB', '2048'))
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001'], supports_credentials=True)

# Reduce Flask logging verbosity
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global variables to store models and results
models = {
    'supervised_model': None,
    'isolation_model': None,
    'encoders': None,
    'scaler': None,
    'threshold_config': None,
    'feature_columns': None
}

# Rate limiting for model status endpoint
last_status_check = 0
STATUS_CHECK_COOLDOWN = int(os.environ.get('STATUS_CHECK_COOLDOWN', '30'))  # seconds between status checks (default 30s)

# Configuration for credit card dataset
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
TARGET_COL = 'Class'  # Credit card dataset uses 'Class' column
ID_COLS = []  # No ID columns in credit card dataset

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_statistics(y_true, y_pred, sup_prob, unsup_score, combined_score):
    """Calculate comprehensive statistics for the analysis results."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Counts
    total_transactions = len(y_true)
    fraudulent_transactions = sum(y_pred)
    legitimate_transactions = total_transactions - fraudulent_transactions
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1Score': float(f1),
        'totalTransactions': int(total_transactions),
        'fraudulentTransactions': int(fraudulent_transactions),
        'legitimateTransactions': int(legitimate_transactions),
        'truePositives': int(tp),
        'falsePositives': int(fp),
        'trueNegatives': int(tn),
        'falseNegatives': int(fn)
    }

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """Get the current status of the fraud detection model."""
    global last_status_check
    
    # Rate limiting to prevent log spam
    current_time = time.time()
    if current_time - last_status_check < STATUS_CHECK_COOLDOWN:
        # Return cached response without logging
        if models['supervised_model'] is not None:
            status = 'ready'
            message = 'Model is ready for analysis'
        else:
            status = 'idle'
            message = 'Model needs to be trained with data'
        
        return jsonify({
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    # Update last check time and log this request (but only occasionally)
    last_status_check = current_time
    
    if models['supervised_model'] is not None:
        status = 'ready'
        message = 'Model is ready for analysis'
    else:
        status = 'idle'
        message = 'Model needs to be trained with data'
    
    return jsonify({
        'status': status,
        'message': message,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/batch-analysis', methods=['POST'])
def batch_analysis():
    """Perform batch analysis on uploaded CSV file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400
        
        # Get sample size if provided
        sample_size = request.form.get('sample_size')
        if sample_size:
            try:
                sample_size = int(sample_size)
            except ValueError:
                return jsonify({'error': 'Invalid sample size'}), 400
        
        # Save uploaded file to a unique temporary path to avoid race conditions
        original_filename = secure_filename(file.filename)
        fd, tmp_path = tempfile.mkstemp(prefix='upload_', suffix='.csv', dir=UPLOAD_FOLDER)
        os.close(fd)
        file.save(tmp_path)
        
        # Run the fraud detection pipeline using your notebook code
        print(f"Starting batch analysis for file: {original_filename}")
        
        try:
            # Load and process the data using your notebook functions
            df = pd.read_csv(tmp_path)
            
            # Clean duplicates
            df.drop_duplicates(inplace=True)
            
            # Feature selection based on correlation with target
            if TARGET_COL in df.columns:
                selected_features = df.corr()[TARGET_COL][:-1].abs().sort_values().tail(14)
                df_selected = selected_features.to_frame().reset_index()
                selected_features = df_selected['index']
                
                # Scale amount column if it exists
                if 'Amount' in df.columns:
                    amount = df['Amount'].values.reshape(-1, 1)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    amount_scaled = scaler.fit_transform(amount)
                    df['Amount'] = amount_scaled
                
                X = df[selected_features]
                y = df[TARGET_COL]
            else:
                X = df.drop(columns=[TARGET_COL])
                y = df[TARGET_COL]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Apply balancing techniques
            print("Applying Random OverSampler...")
            X_train_ros, y_train_ros = fraud_detection.balancedWithRandomOverSampler(X_train, y_train)
            
            print("Applying SMOTE...")
            X_train_smote, y_train_smote = fraud_detection.balanceWithSMOTE(X_train, y_train)
            
            # Train models
            print("Training LightGBM model...")
            import lightgbm as lgb
            model = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
            model.fit(X_train_ros, y_train_ros)
            
            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate statistics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            statistics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1Score': float(f1),
                'totalTransactions': int(len(y_test)),
                'fraudulentTransactions': int(sum(y_pred)),
                'legitimateTransactions': int(len(y_test) - sum(y_pred)),
                'truePositives': int(tp),
                'falsePositives': int(fp),
                'trueNegatives': int(tn),
                'falseNegatives': int(fn)
            }
            
            # Prepare flagged transactions
            flagged_indices = np.where(y_pred == 1)[0]
            flagged_transactions = []
            
            for idx in flagged_indices:
                tx_data = df.iloc[idx]
                flagged_transactions.append({
                    'Transaction_ID': f'TX_{idx}',
                    'User_ID': f'USER_{idx}',
                    'Transaction_Amount': float(tx_data.get('Amount', 0)),
                    'suspicion_score': float(y_prob[idx])
                })
            
            # Store model for real-time analysis
            models['supervised_model'] = model
            models['feature_columns'] = X.columns.tolist()
            
            return jsonify({
                'success': True,
                'statistics': statistics,
                'flaggedTransactions': flagged_transactions,
                'totalFlagged': len(flagged_transactions),
                'analysisTimestamp': datetime.now().isoformat(),
                'fileName': original_filename
            })
            
        finally:
            # Ensure temporary file is removed even if processing fails
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception as cleanup_err:
                print(f"Warning: failed to remove temp file: {cleanup_err}")
        
    except Exception as e:
        print(f"Error in batch analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/real-time-analysis', methods=['POST'])
def real_time_analysis():
    """Perform real-time analysis on a single transaction."""
    try:
        if models['supervised_model'] is None:
            return jsonify({'error': 'Model not trained. Please run batch analysis first.'}), 400
        
        # Get transaction data
        transaction_data = request.json
        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Build dataframe with the expected feature columns, fill missing with None/0
        df = pd.DataFrame([transaction_data])
        
        # Ensure all expected columns exist
        feature_cols = models.get('feature_columns') or []
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        # Drop target if accidentally provided
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
        # Reorder
        if feature_cols:
            df = df[feature_cols]
        
        # Encode categorical features using the same encoders
        for col, le in (models.get('encoders') or {}).items():
            if col == '_target':
                continue
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('')
                known = set(le.classes_)
                # Map unseen to a placeholder (the last index after adding)
                if not df[col].isin(known).all():
                    # Extend classes to include unseen values
                    unseen_values = sorted(set(df[col]) - known)
                    if unseen_values:
                        le.classes_ = np.concatenate([le.classes_, np.array(unseen_values, dtype=le.classes_.dtype)])
                df[col] = le.transform(df[col])
        
        X = df
        
        # Supervised probability
        sup_prob = models['supervised_model'].predict_proba(X)[:, 1] if hasattr(models['supervised_model'], "predict_proba") else models['supervised_model'].predict(X)
        
        # Unsupervised path: add sup_prob and scale with the fitted scaler
        X_unsup = X.copy()
        X_unsup['sup_prob'] = sup_prob
        if models.get('scaler') is not None:
            X_scaled = models['scaler'].transform(X_unsup)
        else:
            X_scaled = X_unsup.values
        raw_unsup = -models['isolation_model'].decision_function(X_scaled)
        unsup_score = (raw_unsup - raw_unsup.min()) / (raw_unsup.max() - raw_unsup.min() + 1e-9)
        
        # Normalize sup prob for combination
        sp = (sup_prob - sup_prob.min()) / (sup_prob.max() - sup_prob.min() + 1e-9)
        alpha = models['threshold_config']['alpha']
        combined_score = alpha * sp + (1 - alpha) * unsup_score
        
        fraud_probability = float(combined_score[0])
        if fraud_probability >= 0.8:
            risk_level = 'high'
            recommendation = 'Block transaction immediately'
        elif fraud_probability >= 0.5:
            risk_level = 'medium'
            recommendation = 'Additional verification required'
        else:
            risk_level = 'low'
            recommendation = 'Transaction appears legitimate'
        
        return jsonify({
            'success': True,
            'fraud_probability': fraud_probability,
            'confidence': 0.85,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'supervised_score': float(sup_prob[0]),
            'anomaly_score': float(unsup_score[0]),
            'combined_score': float(combined_score[0]),
            'analysisTimestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in real-time analysis: {str(e)}")
        return jsonify({'error': f'Real-time analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': models['supervised_model'] is not None
    })

@app.route('/api/debug/rate-limit', methods=['GET'])
def debug_rate_limit():
    """Debug endpoint to check rate limiting status."""
    global last_status_check
    current_time = time.time()
    time_since_last = current_time - last_status_check
    return jsonify({
        'last_status_check': last_status_check,
        'current_time': current_time,
        'time_since_last': time_since_last,
        'cooldown_remaining': max(0, STATUS_CHECK_COOLDOWN - time_since_last),
        'rate_limited': time_since_last < STATUS_CHECK_COOLDOWN
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size is {MAX_CONTENT_MB}MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Main execution moved to main.py