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
import threading
import random
import uuid

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

# Set up detailed logging for our application
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Real-time simulation variables
simulation_running = False
simulation_thread = None
simulation_transactions = []
SIMULATION_FILE = 'realtime_transactions.json'

# Configuration for credit card dataset
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
TARGET_COL = 'Class'  # Credit card dataset uses 'Class' column
ID_COLS = []  # No ID columns in credit card dataset

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_dummy_transaction():
    """Generate a dummy transaction with realistic credit card fraud dataset features."""
    # Generate V1-V28 features (PCA transformed features in credit card dataset)
    transaction = {}
    
    # Generate V1-V28 (PCA features) - normally distributed around 0
    for i in range(1, 29):
        transaction[f'V{i}'] = np.random.normal(0, 1)
    
    # Generate Time (seconds elapsed since first transaction)
    transaction['Time'] = time.time()
    
    # Generate Amount (transaction amount) - log-normal distribution for realistic amounts
    transaction['Amount'] = max(0.01, np.random.lognormal(3, 1.5))  # Mean around $20-50, some large amounts
    
    # Add transaction metadata
    transaction['transaction_id'] = str(uuid.uuid4())
    transaction['timestamp'] = datetime.now().isoformat()
    
    # Occasionally create suspicious patterns (higher fraud likelihood)
    if random.random() < 0.15:  # 15% chance of suspicious pattern
        # Large amount late at night
        transaction['Amount'] = random.uniform(500, 2000)
        transaction['V1'] = random.uniform(-3, -1)  # Suspicious pattern
        transaction['V2'] = random.uniform(2, 4)
        transaction['V3'] = random.uniform(-3, -1)
    
    return transaction

def simulation_worker():
    """Background worker that generates transactions every second."""
    global simulation_running, simulation_transactions
    
    while simulation_running:
        try:
            # Generate new transaction
            transaction = generate_dummy_transaction()
            simulation_transactions.append(transaction)
            
            # Keep only last 1000 transactions to prevent memory issues
            if len(simulation_transactions) > 1000:
                simulation_transactions = simulation_transactions[-1000:]
            
            # Save to JSON file
            with open(SIMULATION_FILE, 'w') as f:
                json.dump(simulation_transactions, f, indent=2)
            
            logger.info(f"Generated transaction {len(simulation_transactions)}: Amount=${transaction['Amount']:.2f}")
            
            # Wait 1 second
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in simulation worker: {str(e)}")
            break

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
        if models['supervised_model'] is not None and models['feature_columns'] is not None:
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
    
    if models['supervised_model'] is not None and models['feature_columns'] is not None:
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
        logger.info(f"Starting batch analysis for file: {original_filename}")
        
        try:
            # Load and process the data using your notebook functions
            df = pd.read_csv(tmp_path)
            
            # Display dataset info
            logger.info("Dataset Info:")
            logger.info(f"Dataset Shape: {df.shape}")
            logger.info(f"Dataset Columns: {list(df.columns)}")
            logger.info(f"Dataset Types: {df.dtypes.to_dict()}")
            
            # Display initial transaction counts
            if TARGET_COL in df.columns:
                non_fraudulent_count = df[TARGET_COL].value_counts()[0]
                fraudulent_count = df[TARGET_COL].value_counts()[1]
                logger.info(f"Initial Transaction Distribution:")
                logger.info(f"Number of normal transactions = {non_fraudulent_count} (% {non_fraudulent_count/len(df)*100})")
                logger.info(f"Number of fraudulent transactions = {fraudulent_count} (% {fraudulent_count/len(df)*100})")
            
            # Clean duplicates
            df.drop_duplicates(inplace=True)
            logger.info(f"After removing duplicates: {df.shape}")
            
            # Display transaction counts after cleaning
            if TARGET_COL in df.columns:
                non_fraudulent_count = df[TARGET_COL].value_counts()[0]
                fraudulent_count = df[TARGET_COL].value_counts()[1]
                logger.info(f"After Cleaning:")
                logger.info(f"Number of normal transactions = {non_fraudulent_count} (% {non_fraudulent_count/len(df)*100})")
                logger.info(f"Number of fraudulent transactions = {fraudulent_count} (% {fraudulent_count/len(df)*100})")
            
            # Feature selection based on correlation with target
            if TARGET_COL in df.columns:
                logger.info(f"Feature Selection:")
                selected_features = df.corr()[TARGET_COL][:-1].abs().sort_values().tail(14)
                df_selected = selected_features.to_frame().reset_index()
                selected_features = df_selected['index']
                logger.info(f"Selected {len(selected_features)} features: {list(selected_features)}")
                
                # Scale amount column if it exists
                if 'Amount' in df.columns:
                    logger.info("Scaling Amount column...")
                    amount = df['Amount'].values.reshape(-1, 1)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    amount_scaled = scaler.fit_transform(amount)
                    df['Amount'] = amount_scaled
                
                X = df[selected_features]
                y = df[TARGET_COL]
                logger.info(f"Feature matrix shape: {X.shape}")
                logger.info(f"Target vector shape: {y.shape}")
            else:
                X = df.drop(columns=[TARGET_COL])
                y = df[TARGET_COL]
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=85, stratify=y)
            logger.info(f"Data Split:")
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Training labels: {y_train.value_counts().to_dict()}")
            logger.info(f"Test labels: {y_test.value_counts().to_dict()}")
            
            # Apply balancing techniques
            logger.info("Applying SMOTE...")
            X_train_smote, y_train_smote = fraud_detection.balanceWithSMOTE(X_train, y_train)
            logger.info(f"After SMOTE - Training set: {X_train_smote.shape}")
            logger.info(f"After SMOTE - Training labels: {y_train_smote.value_counts().to_dict()}")
            
            # Train models
            logger.info("Training LightGBM model...")
            import lightgbm as lgb
            model = lgb.LGBMClassifier()  # Using default parameters like in original notebook
            logger.info(f"Model parameters: {model.get_params()}")
            model.fit(X_train_smote, y_train_smote)
            logger.info("LightGBM training completed!")
            
            # Get predictions
            logger.info("Making predictions...")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            logger.info(f"Predictions shape: {y_pred.shape}")
            logger.info(f"Prediction distribution: {np.bincount(y_pred)}")
            
            # Calculate statistics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
            auc = roc_auc_score(y_test, y_pred)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            logger.info(f"Model Performance:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"AUC: {auc:.4f}")
            logger.info(f"Confusion Matrix:")
            logger.info(f"  True Negatives: {tn}")
            logger.info(f"  False Positives: {fp}")
            logger.info(f"  False Negatives: {fn}")
            logger.info(f"  True Positives: {tp}")
            
            # Get actual test set distribution
            actual_fraudulent = int(sum(y_test))
            actual_legitimate = int(len(y_test) - sum(y_test))
            
            logger.info(f"Test Set Distribution:")
            logger.info(f"  Actual Fraudulent: {actual_fraudulent}")
            logger.info(f"  Actual Legitimate: {actual_legitimate}")
            logger.info(f"  Predicted Fraudulent: {int(sum(y_pred))}")
            logger.info(f"  Predicted Legitimate: {int(len(y_test) - sum(y_pred))}")
            
            statistics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1Score': float(f1),
                'auc': float(auc),
                'totalTransactions': int(len(y_test)),
                'fraudulentTransactions': actual_fraudulent,  # Actual frauds in test set
                'legitimateTransactions': actual_legitimate,  # Actual legitimate in test set
                'truePositives': int(tp),
                'falsePositives': int(fp),
                'trueNegatives': int(tn),
                'falseNegatives': int(fn)
            }
            
            # Prepare flagged transactions with complete original data
            logger.info(f"Preparing flagged transactions...")
            flagged_indices = np.where(y_pred == 1)[0]
            flagged_transactions = []
            
            # IMPORTANT: Create a copy of the original dataframe BEFORE any modifications
            original_complete_df = df.copy()
            logger.info(f"Original complete dataframe columns: {list(original_complete_df.columns)}")
            
            # Get the actual indices from the test set that correspond to the original dataframe
            test_indices = X_test.index
            logger.info(f"Test set indices: {test_indices[:10]} (first 10)")
            logger.info(f"Test set shape: {X_test.shape}")
            logger.info(f"Original dataframe shape: {original_complete_df.shape}")
            
            # Debug: Log the columns we're working with
            logger.info(f"Original complete dataframe columns: {list(original_complete_df.columns)}")
            
            for idx in flagged_indices:
                try:
                    # Get the actual index in the original dataframe
                    actual_idx = test_indices[idx]
                    logger.info(f"Processing flagged transaction {idx}: mapped to original index {actual_idx}")
                    
                    # Get the original transaction data with all columns
                    # Use .loc because test indices are label-based, not positional
                    original_tx = original_complete_df.loc[actual_idx]
                    tx_data = X_test.iloc[idx]
                    
                    # Create transaction object with all original columns
                    transaction = {}
                    
                    # Add all original columns
                    for col in original_complete_df.columns:
                        transaction[col] = original_tx[col]
                    
                    # Add fraud detection specific fields
                    transaction['suspicion_score'] = float(y_prob[idx])
                    transaction['predicted_class'] = 'Fraudulent'
                    transaction['model_confidence'] = float(y_prob[idx])
                    
                    flagged_transactions.append(transaction)
                    
                except Exception as e:
                    logger.error(f"Error processing transaction {idx}: {str(e)}")
                    logger.error(f"test_indices length: {len(test_indices)}, flagged_indices: {flagged_indices}")
                    logger.error(f"X_test.index: {X_test.index}")
                    continue
            
            # Debug: Log what we're actually sending
            if flagged_transactions:
                logger.info(f"First flagged transaction keys: {list(flagged_transactions[0].keys())}")
                logger.info(f"Sample transaction data: {flagged_transactions[0]}")
            
            logger.info(f"Total predicted frauds (flagged): {len(flagged_transactions)}")
            logger.info(f"True Positives (correctly identified frauds): {tp}")
            logger.error(f"False Positives (incorrectly flagged): {fp}")
            logger.info(f"Analysis completed successfully!")
            
            logger.info(f"Total predicted frauds (flagged): {len(flagged_transactions)}")
            logger.info(f"True Positives (correctly identified frauds): {tp}")
            logger.info(f"False Positives (incorrectly flagged): {fp}")
            logger.info(f"Analysis completed successfully!")
            
            # Store model for real-time analysis
            models['supervised_model'] = model
            models['feature_columns'] = X.columns.tolist()
            
            return jsonify({
                'success': True,
                'statistics': statistics,
                'flaggedTransactions': flagged_transactions,
                'totalFlagged': int(tp),  # Show only true positives (correctly identified frauds)
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
        
        fraud_probability = float(sup_prob[0])
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
            'analysisTimestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in real-time analysis: {str(e)}")
        return jsonify({'error': f'Real-time analysis failed: {str(e)}'}), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start the real-time transaction simulation."""
    global simulation_running, simulation_thread, simulation_transactions
    
    try:
        if simulation_running:
            return jsonify({'error': 'Simulation is already running'}), 400
        
        # Clear previous transactions
        simulation_transactions = []
        if os.path.exists(SIMULATION_FILE):
            os.remove(SIMULATION_FILE)
        
        # Start simulation
        simulation_running = True
        simulation_thread = threading.Thread(target=simulation_worker, daemon=True)
        simulation_thread.start()
        
        logger.info("Real-time transaction simulation started")
        
        return jsonify({
            'success': True,
            'message': 'Simulation started successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error starting simulation: {str(e)}")
        return jsonify({'error': f'Failed to start simulation: {str(e)}'}), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop the real-time transaction simulation."""
    global simulation_running, simulation_thread
    
    try:
        if not simulation_running:
            return jsonify({'error': 'Simulation is not running'}), 400
        
        simulation_running = False
        
        # Wait for thread to finish (with timeout)
        if simulation_thread and simulation_thread.is_alive():
            simulation_thread.join(timeout=2)
        
        # Load transactions from file to ensure we have the latest count
        if os.path.exists(SIMULATION_FILE):
            try:
                with open(SIMULATION_FILE, 'r') as f:
                    file_transactions = json.load(f)
                    # Update global variable with file data
                    global simulation_transactions
                    simulation_transactions = file_transactions
            except Exception as e:
                logger.warning(f"Could not load transactions from file: {str(e)}")
        
        logger.info(f"Real-time transaction simulation stopped. Total transactions: {len(simulation_transactions)}")
        
        return jsonify({
            'success': True,
            'message': 'Simulation stopped successfully',
            'timestamp': datetime.now().isoformat(),
            'total_transactions': len(simulation_transactions)
        })
        
    except Exception as e:
        logger.error(f"Error stopping simulation: {str(e)}")
        return jsonify({'error': f'Failed to stop simulation: {str(e)}'}), 500

@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get the current status of the simulation."""
    global simulation_transactions
    
    # If global variable is empty but file exists, try to load from file
    if not simulation_transactions and os.path.exists(SIMULATION_FILE):
        try:
            with open(SIMULATION_FILE, 'r') as f:
                file_transactions = json.load(f)
                simulation_transactions = file_transactions
                logger.info(f"Loaded {len(simulation_transactions)} transactions from file")
        except Exception as e:
            logger.warning(f"Could not load transactions from file: {str(e)}")
    
    return jsonify({
        'running': simulation_running,
        'total_transactions': len(simulation_transactions),
        'latest_transactions': simulation_transactions[-5:] if simulation_transactions else [],
        'file_exists': os.path.exists(SIMULATION_FILE),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/simulation/analyze', methods=['POST'])
def analyze_simulation_data():
    """Analyze the simulated transactions using the current model."""
    global simulation_transactions
    
    try:
        if models['supervised_model'] is None:
            return jsonify({'error': 'Model not trained. Please run batch analysis first to train a model.'}), 400
        
        if not simulation_transactions:
            return jsonify({'error': 'No simulation data available. Start simulation first.'}), 400
        
        # Convert transactions to DataFrame
        df = pd.DataFrame(simulation_transactions)
        
        # Prepare features for the model (same features used in training)
        feature_cols = models.get('feature_columns', [])
        if not feature_cols:
            return jsonify({'error': 'No feature columns available from trained model.'}), 400
        
        # Ensure all required columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select only the features used in training
        X = df[feature_cols]
        
        # Get predictions
        predictions = models['supervised_model'].predict(X)
        probabilities = models['supervised_model'].predict_proba(X)[:, 1]
        
        # Prepare results
        flagged_transactions = []
        for i, (idx, row) in enumerate(df.iterrows()):
            if predictions[i] == 1:  # Flagged as fraud
                transaction = row.to_dict()
                transaction['fraud_probability'] = float(probabilities[i])
                transaction['predicted_class'] = 'Fraudulent'
                transaction['confidence'] = float(probabilities[i])
                flagged_transactions.append(transaction)
        
        # Calculate basic statistics
        total_transactions = len(simulation_transactions)
        flagged_count = int(sum(predictions))
        fraud_rate = (flagged_count / total_transactions * 100) if total_transactions > 0 else 0
        
        statistics = {
            'totalTransactions': total_transactions,
            'flaggedTransactions': flagged_count,
            'fraudRate': fraud_rate,
            'averageFraudProbability': float(np.mean(probabilities)),
            'maxFraudProbability': float(np.max(probabilities)),
            'minFraudProbability': float(np.min(probabilities))
        }
        
        logger.info(f"Analyzed {total_transactions} simulated transactions, flagged {flagged_count} as potential fraud")
        
        return jsonify({
            'success': True,
            'statistics': statistics,
            'flaggedTransactions': flagged_transactions,
            'totalFlagged': flagged_count,
            'analysisTimestamp': datetime.now().isoformat(),
            'dataSource': 'real-time-simulation'
        })
        
    except Exception as e:
        logger.error(f"Error analyzing simulation data: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/export-fraudulent-transactions', methods=['POST'])
def export_fraudulent_transactions():
    """Export fraudulent transactions to CSV file."""
    try:
        # Get the flagged transactions from the request
        data = request.json
        if not data or 'flaggedTransactions' not in data:
            return jsonify({'error': 'No flagged transactions data provided'}), 400
        
        flagged_transactions = data['flaggedTransactions']
        if not flagged_transactions:
            return jsonify({'error': 'No fraudulent transactions to export'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(flagged_transactions)
        
        # Create a temporary file
        fd, tmp_path = tempfile.mkstemp(prefix='fraudulent_transactions_', suffix='.csv')
        os.close(fd)
        
        # Save to CSV
        df.to_csv(tmp_path, index=False)
        
        # Return the file
        return send_file(
            tmp_path,
            as_attachment=True,
            download_name='fraudulent_transactions_detected.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting fraudulent transactions: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

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

@app.route('/api/debug/simulation', methods=['GET'])
def debug_simulation():
    """Debug endpoint to check simulation state."""
    global simulation_transactions
    return jsonify({
        'simulation_running': simulation_running,
        'simulation_transactions_count': len(simulation_transactions),
        'file_exists': os.path.exists(SIMULATION_FILE),
        'file_size': os.path.getsize(SIMULATION_FILE) if os.path.exists(SIMULATION_FILE) else 0,
        'latest_transactions': simulation_transactions[-3:] if simulation_transactions else [],
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size is {MAX_CONTENT_MB}MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Main execution moved to main.py