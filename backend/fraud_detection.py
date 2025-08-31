import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import math
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import classification_report
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")

# for balancing dataset
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


def displayValues(y_train):
    print('Non-Frauds:', y_train.value_counts()[0], '/', round(y_train.value_counts()[0]/len(y_train) * 100,2), '% of the dataset')
    print('Frauds:', y_train.value_counts()[1], '/',round(y_train.value_counts()[1]/len(y_train) * 100,2), '% of the dataset')

def balancedWithRandomOverSampler(X_train, y_train):
    ros = RandomOverSampler(random_state=50)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    
    displayValues(y_train_ros)
    
    return X_train_ros, y_train_ros


def balanceWithSMOTE(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    displayValues(y_train_smote)
    
    return X_train_smote, y_train_smote


#Machine learning models

all_performances = pd.DataFrame()
list_clf_name = []
list_pred = []
list_model = []

def fit_model(model, X_train, y_train):
    X_model = model.fit(X_train,y_train)
    return X_model

def add_list(name, model, y_pred):
    global list_clf_name, list_pred, list_model, list_x_test
    list_clf_name.append(name)
    list_model.append(model)
    list_pred.append(y_pred)


def add_all_performances(name, precision, recall, f1_score, AUC):
    global all_performances
    models = pd.DataFrame([[name, precision, recall, f1_score, AUC]],
                         columns=["model_name","precision", "recall", "f1_score", "AUC"])
    all_performances = pd.concat([all_performances, models], ignore_index=True)  

    all_performances= all_performances.drop_duplicates()
      
    
def calculate_scores(X_train, X_test, y_train, y_test, y_pred, name, model):
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = mt.precision_score(y_test,y_pred)
    recall = mt.recall_score(y_test,y_pred)
    f1_score= mt.f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    
    add_list(name, model, y_pred)
    add_all_performances(name, precision, recall, f1_score, AUC)
    #print(all_performances.sort_values(by=['f1_score'], ascending=False))
    

def model_performance(model, X_train, X_test, y_train, y_test, technique_name):

    name= model.__class__.__name__+"_"+technique_name
    x_model = fit_model(model, X_train, y_train)
    y_pred = x_model.predict(X_test)
    print("***** "+ name +" DONE *****")

    calculate_scores(X_train, X_test, y_train, y_test, y_pred, name, model)
    


def display_all_confusion_matrices(y_test):
    column = 2
    total_models = all_performances["model_name"].count()
    row = max(1, int(np.ceil(total_models / column)))   # row kabhi 0 nahi hoga

    f, ax = plt.subplots(row, column, figsize=(20, 5*row), sharey='row')
    ax = ax.flatten()

    for i in range(total_models):
        cf_matrix = confusion_matrix(y_test, list_pred[i])
        disp = ConfusionMatrixDisplay(cf_matrix)
        disp.plot(ax=ax[i], xticks_rotation=45)
        disp.ax_.set_title(list_clf_name[i]+"\nAccuracy:{accuracy:.4f}\nAUC:{auc:.4f}"
                           .format(accuracy= accuracy_score(y_test, list_pred[i]),
                                   auc= roc_auc_score(y_test, list_pred[i])),
                             fontsize=14)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i != 0:
            disp.ax_.set_ylabel('')

    # f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust()
    f.colorbar(disp.im_)
    plt.show()


def show_graphs(df):
    fraud_counts = df['Class'].value_counts()

    # pie chart
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))

    ax1.pie(fraud_counts, labels=['Non-fraudulent', 'Fraudulent'], colors=['lightblue', 'red'], autopct='%1.1f%%')
    ax1.set_title('Transaction Class Distribution')

    # bar plot
    ax2.bar(['Non-fraudulent', 'Fraudulent'], fraud_counts.values, color=['lightblue', 'red'])
    ax2.set_xlabel('Transaction Class')
    ax2.set_ylabel('Number of Transactions')
    ax2.set_title('Credit Card Fraud Detection')
    for i in ax2.containers:
        ax2.bar_label(i,)

    plt.show()


def boxplots_custom(dataset, columns_list, rows, cols, suptitle):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(16,25))
    fig.suptitle(suptitle,y=1, size=25)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        sns.boxplot(data=dataset[data], orient='h', ax=axs[i])
        axs[i].set_title(data + ', skewness is: '+str(round(dataset[data].skew(axis = 0, skipna = True),2)))
        
    plt.tight_layout()


def plot_histograms(dataset, columns, cols_per_row=3, bins=60):
    n = len(columns)
    rows = math.ceil(n / cols_per_row)
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row*5, rows*4))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        axes[i].hist(dataset[col], bins=bins, linewidth=0.5, edgecolor="white")
        axes[i].set_title(f"{col} distribution")
    
    # Hide unused subplots (if any)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def find_interquartile_range(df, column_name):
    try:
        Q1 = df[column_name].quantile(0.25)  # First quartile
        Q3 = df[column_name].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range

        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
    except TypeError:
        pass
        
    return lower_limit, upper_limit

def separate_outliers(df, column_name):
    try:
        lower_limit, upper_limit = find_interquartile_range(df, column_name)
        outlier_mask_lower = df[column_name] < lower_limit
        outlier_mask_upper = df[column_name] > upper_limit
        
        if lower_limit == upper_limit == 0:
            df[column_name][outlier_mask_lower] = 0   # (?)
            df[column_name][outlier_mask_upper] = 1
        else:
            df[column_name][outlier_mask_lower] = lower_limit  
            df[column_name][outlier_mask_upper] = upper_limit
        
        # print("separate_outliers worked!")
        print("Lower limit: ", lower_limit)
        print("Upper limit: ", upper_limit)
        # print("Unique: ", df[column_name].unique())

    except:
        print("separate_outliers error!")
        pass


# Main execution code (this will run when the file is imported)
def main():
    # "Importing the Dataset" 
    data = pd.read_csv('../dataset/creditcard.csv')
    data.head()

    df = data.copy()

    df.info() #We don't have any null values

    df.describe().T

    # Our dataset is quite unbalanced
    show_graphs(df)

    non_fradulent_count = df['Class'].value_counts()[0]
    fradulent_count = df['Class'].value_counts()[1]

    print(f"Number of normal transactions = {non_fradulent_count} (% {non_fradulent_count/len(df)*100})")
    print(f"Number of fraudulent transactions = {fradulent_count} (% {fradulent_count/len(df)*100})")

    plt.figure(figsize = (40,20))
    sns.heatmap(df.corr(), cmap="magma_r", annot=True);

    df.drop_duplicates(inplace=True)   #We cleaned the duplicate data

    numeric_columns = (list(df.loc[:, 'V1':'Amount']))

    boxplots_custom(dataset=df, columns_list=numeric_columns, rows=10, cols=3, suptitle='')
    plt.tight_layout()

    # Call function
    plot_histograms(df, numeric_columns)

    show_graphs(df)

    non_fraudulent_count = df['Class'].value_counts()[0]
    fraudulent_count = df['Class'].value_counts()[1]

    print(f"Number of normal transactions = {non_fraudulent_count} (% {non_fraudulent_count/len(df)*100})")
    print(f"Number of fraudulent transactions = {fraudulent_count} (% {fraudulent_count/len(df)*100})")

    df["Class"].unique()

    # Feature Selection

    plt.figure(figsize=(15,8))
    d = df.corr()['Class'][:-1].abs().sort_values().plot(kind='bar', title='Most important features')

    plt.show()

    #selected_features = (df.corr()['Class'][:-1].abs() > 0.15)
    selected_features = df.corr()['Class'][:-1].abs().sort_values().tail(14)
    df_selected= selected_features.to_frame().reset_index()
    selected_featues = df_selected['index']

    sns.jointplot(x='V11', y='V3',hue='Class', data=df, palette = 'dark')

    amount = df['Amount'].values.reshape(-1, 1)

    scaler = StandardScaler()
    amount_scaled = scaler.fit_transform(amount)

    df['Amount'] = amount_scaled

    X = df[selected_featues]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.3, random_state = 85)

    ### Random Oversampling

    # Data augmentation through sampling may cause overfitting.
    # balancedWithRandomOverSampler:  it is a funcion
    X_train_ros, y_train_ros = balancedWithRandomOverSampler(X_train,y_train)

    ### SMOTE

    # SMOTE (Synthetic Minority Oversampling Technique) synthesizes samples for the minority class. 
    # SMOTE works by selecting samples that are close in the feature space, drawing a line between them, 
    # and generating a new sample at a point along that line.

    X_train_smote, y_train_smote = balanceWithSMOTE(X_train, y_train)

    # Model Building and Training

    ml_models = [ lgb.LGBMClassifier()] 
                 

    for i in ml_models:
        model_performance(i, X_train_ros, X_test, y_train_ros, y_test, "RandomOverSampler")
        model_performance(i, X_train_smote, X_test, y_train_smote, y_test, "SMOTE")

    all_performances.sort_values(by=['f1_score','AUC'], ascending=False)

    # Comparison of Performances

    display_all_confusion_matrices(y_test)

if __name__ == "__main__":
    main()

# Flask compatibility functions
def load_df(file_path, target_col='Class', sample_n=None, id_cols=None):
    """Load and prepare dataframe for analysis."""
    try:
        df = pd.read_csv(file_path)
        
        # Sample if requested
        if sample_n and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        
        # Ensure target column exists
        if target_col not in df.columns:
            # Try to find similar column names
            possible_targets = [col for col in df.columns if 'fraud' in col.lower() or 'class' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
            else:
                # Create a dummy target column for demonstration
                df[target_col] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
        
        return df, target_col
    except Exception as e:
        print(f"Error loading dataframe: {e}")
        return None, None

def preprocess_for_models(df, target_col, id_cols=None):
    """Preprocess dataframe for model training."""
    try:
        # Remove ID columns if specified
        if id_cols:
            df = df.drop(columns=[col for col in id_cols if col in df.columns])
        
        # Clean duplicates
        df.drop_duplicates(inplace=True)
        
        # Feature selection based on correlation with target
        if target_col in df.columns:
            selected_features = df.corr()[target_col][:-1].abs().sort_values().tail(14)
            df_selected = selected_features.to_frame().reset_index()
            selected_features = df_selected['index']
            
            # Scale amount column if it exists
            if 'Amount' in df.columns:
                amount = df['Amount'].values.reshape(-1, 1)
                scaler = StandardScaler()
                amount_scaled = scaler.fit_transform(amount)
                df['Amount'] = amount_scaled
            
            X = df[selected_features]
            y = df[target_col]
        else:
            X = df.drop(columns=[target_col])
            y = df[target_col]
        
        return X, y, None, None, X.columns.tolist()
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None, None

def supervised_scores(X_train, X_test, y_train, y_test):
    """Train supervised model and return scores."""
    try:
        # Use LightGBM as the supervised model
        model = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
        model.fit(X_train, y_train)
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return model, y_pred, y_prob
    except Exception as e:
        print(f"Error in supervised training: {e}")
        return None, None, None

def unsupervised_scores(X_train, X_test, y_train, y_test):
    """Train unsupervised model and return scores."""
    try:
        # Use Isolation Forest for anomaly detection
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(random_state=42, contamination=0.1)
        model.fit(X_train)
        
        # Get anomaly scores (negative values indicate anomalies)
        scores = -model.decision_function(X_test)
        
        # Normalize scores to 0-1 range
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        
        return model, scores
    except Exception as e:
        print(f"Error in unsupervised training: {e}")
        return None, None

def tune_threshold_and_eval(y_true, sup_prob, unsup_score, alpha_range=np.arange(0.1, 1.0, 0.1)):
    """Tune the combination parameter alpha and evaluate."""
    best_alpha = 0.5
    best_f1 = 0
    
    for alpha in alpha_range:
        combined_score = alpha * sup_prob + (1 - alpha) * unsup_score
        threshold = np.percentile(combined_score, 90)  # Use 90th percentile as threshold
        y_pred = (combined_score >= threshold).astype(int)
        
        f1 = mt.f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
    
    # Get best predictions
    best_combined = best_alpha * sup_prob + (1 - best_alpha) * unsup_score
    best_threshold = np.percentile(best_combined, 90)
    best_pred = (best_combined >= best_threshold).astype(int)
    
    return {
        'alpha': best_alpha,
        'threshold': best_threshold,
        'pred': best_pred,
        'combined': best_combined,
        'f1_score': best_f1
    }

def gnn_baseline_flagging(df, target_col):
    """Simple baseline flagging using basic heuristics."""
    # This is a placeholder for GNN-based flagging
    # For now, we'll use a simple rule-based approach
    flags = np.zeros(len(df))
    
    # Flag high amount transactions
    amount_col = [col for col in df.columns if 'amount' in col.lower()]
    if amount_col:
        amount_threshold = df[amount_col[0]].quantile(0.95)
        flags[df[amount_col[0]] > amount_threshold] = 1
    
    return flags

def run_chained_pipeline(file_path, target_col='Class', sample_n=None, id_cols=None):
    """Run the complete fraud detection pipeline."""
    try:
        print("Loading data...")
        # Load data
        df, target_col = load_df(file_path, target_col, sample_n, id_cols)
        if df is None:
            return None
        
        print("Preprocessing data...")
        # Preprocess
        X, y, encoders, scaler, feature_columns = preprocess_for_models(df, target_col, id_cols)
        if X is None:
            return None
        
        print("Splitting data...")
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Apply balancing techniques
        print("Applying Random OverSampler...")
        X_train_ros, y_train_ros = balancedWithRandomOverSampler(X_train, y_train)
        
        print("Applying SMOTE...")
        X_train_smote, y_train_smote = balanceWithSMOTE(X_train, y_train)
        
        print("Training supervised model with Random OverSampler...")
        # Train supervised model with Random OverSampler
        sup_model_ros, sup_pred_ros, sup_prob_ros = supervised_scores(X_train_ros, X_test, y_train_ros, y_test)
        if sup_model_ros is None:
            return None
        
        print("Training supervised model with SMOTE...")
        # Train supervised model with SMOTE
        sup_model_smote, sup_pred_smote, sup_prob_smote = supervised_scores(X_train_smote, X_test, y_train_smote, y_test)
        if sup_model_smote is None:
            return None
        
        print("Training unsupervised model...")
        # Train unsupervised model
        iso_model, unsup_score = unsupervised_scores(X_train, X_test, y_train, y_test)
        if iso_model is None:
            return None
        
        print("Tuning model combination...")
        # Combine models (using Random OverSampler results as primary)
        best_config = tune_threshold_and_eval(y_test, sup_prob_ros, unsup_score)
        
        print("Running baseline flagging...")
        # GNN baseline (placeholder)
        gnn_flags = gnn_baseline_flagging(df, target_col)
        
        print("Pipeline completed successfully!")
        return {
            'df': df,
            'y': y_test,
            'sup_model': sup_model_ros,  # Use Random OverSampler model as primary
            'sup_model_smote': sup_model_smote,  # Keep SMOTE model for comparison
            'iso_model': iso_model,
            'encoders': encoders,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'sup_prob': sup_prob_ros,  # Use Random OverSampler probabilities
            'sup_prob_smote': sup_prob_smote,  # Keep SMOTE probabilities for comparison
            'unsup_score': unsup_score,
            'best': best_config,
            'gnn_flags': gnn_flags,
            'X_train_ros': X_train_ros,
            'y_train_ros': y_train_ros,
            'X_train_smote': X_train_smote,
            'y_train_smote': y_train_smote
        }
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        return None
