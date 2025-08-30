# -------------------------
# Chained Pipeline: Supervised -> Unsupervised -> Bandit (threshold) -> GNN-baseline
# Jupyter copy-paste ready
# -------------------------

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# =========== CONFIG ===========
# -------------------------
dataset_path = None  # Will be set when dataset is uploaded
target_col = "Fraudulent"                                      # <<-- set target column
SAMPLE_N = None   # Set None to use full dataset, or e.g. 200000 for faster runs
RANDOM_STATE = 42
# id_cols=['nameOrig','nameDest']
id_cols = ["Transaction_ID", "User_ID"]

# -------------------------

# -------------------------
# Helper: safe read + optional sampling
# -------------------------
def load_df(path, sample_n=None, random_state=RANDOM_STATE):
    df = pd.read_csv(path)
    # simple NA clean
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if sample_n is not None and sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return df

# -------------------------
# Preprocessing: label-encode object cols; return X_df (encoded), y, encoders dict
# -------------------------
def preprocess_for_models(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    encoders = {}
    # label-encode object/categorical columns (inplace)
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    # also encode target if object
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
        encoders['_target'] = le
    return X, y, encoders

# -------------------------
# Step 1: Supervised model -> produce sup_prob (probability of fraud)
# -------------------------
def supervised_scores(X, y, random_state=RANDOM_STATE):
    # Use a RandomForest with class_weight to help imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced_subsample', random_state=random_state)
    clf.fit(X_train, y_train)
    # get probabilities for all rows (fit on full X could be done but we'll use test for evaluation; we'll produce prob for full X by predicting on X)
    sup_prob = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.predict(X)
    # evaluate on holdout
    y_test_pred = clf.predict(X_test)
    print("\n--- Supervised (RandomForest) evaluation on holdout ---")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, digits=4))
    return sup_prob, clf

# -------------------------
# Step 2: Unsupervised (IsolationForest) on features + sup_prob -> produce anomaly score
# -------------------------
def unsupervised_scores(X, sup_prob, contamination='auto', random_state=RANDOM_STATE):
    # Build matrix: numeric features (X already numeric after encoding) + supervised probability as extra feature
    X_unsup = X.copy()
    X_unsup['sup_prob'] = sup_prob
    scaler = StandardScaler(with_mean=False)   # with_mean=False to be safe with sparse-ish
    X_scaled = scaler.fit_transform(X_unsup)
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=random_state)
    iso.fit(X_scaled)
    # decision_function: higher = more normal; we invert so higher = more anomalous
    raw_scores = -iso.decision_function(X_scaled)
    # normalize to [0,1]
    scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return scores, iso, scaler

# -------------------------
# Step 3: Combine scores and tune threshold (bandit-like simple sweep) -> find best threshold by F1
# -------------------------



# Disadvantage - isme bhot saare fraud(1843) ko non fraud bta rha h. Advantage - Bhot km(4) non fraud ko fraud bta rha h.
def tune_threshold_and_eval(y_true, sup_prob, unsup_score):
    # Combine using weighted sum (we try multiple weightings)
    best = {'f1':-1}
    # normalize sup_prob to 0-1
    sp = (sup_prob - sup_prob.min()) / (sup_prob.max() - sup_prob.min() + 1e-9)
    us = unsup_score
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        combined = alpha * sp + (1-alpha) * us
        # sweep quantile thresholds from 0.995 down to 0.90
        for q in [0.999,0.998,0.995,0.99,0.98,0.97,0.95,0.90]:
            th = np.quantile(combined, q)
            pred = (combined >= th).astype(int)
            p,r,f,_ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)

            
            # if (p > 0.5 and f > best['f1']):   # precision > 50% ho, tabhi accept kar
            #     best = {'alpha':alpha, 'quantile':q, 'threshold':th, 'f1':f, 'precision':p, 'recall':r, 'pred':pred, 'combined':combined}

            
            if f > best['f1']:
                best = {'alpha':alpha, 'quantile':q, 'threshold':th, 'f1':f, 'precision':p, 'recall':r, 'pred':pred, 'combined':combined}
    # Print best found
    print("\n--- Threshold tuning (simple sweep) ---")
    print(f"Best alpha: {best['alpha']}  quantile: {best['quantile']}  threshold: {best['threshold']:.6f}")
    print(f"Precision: {best['precision']:.4f}  Recall: {best['recall']:.4f}  F1: {best['f1']:.4f}")
    return best


# -------------------------
# Step 4: Graph baseline: aggregate combined suspicion to node (account) and rank
# -------------------------
def gnn_baseline_flagging(df, combined_score, id_cols = ["Transaction_ID", "User_ID"], top_k_accounts=20):
    # combined_score: np.array aligned with df rows
    # create df copy with id cols and score
    tmp = df[id_cols].copy()
    tmp = tmp.assign(combined_score = combined_score)
    # aggregate per node (account): for each id column, map account -> mean score
    account_scores = {}
    for col in id_cols:
        grp = tmp.groupby(col)['combined_score'].mean()
        for acc, sc in grp.items():
            account_scores.setdefault(acc, []).append(sc)
    # average across different id columns if present
    account_mean = {acc: np.mean(scores) for acc, scores in account_scores.items()}
    # top suspicious accounts
    top_accounts = sorted(account_mean.items(), key=lambda x: -x[1])[:top_k_accounts]
    top_accounts_df = pd.DataFrame(top_accounts, columns=['account', 'suspicion_score'])
    # mark transactions where either origin or dest in top accounts
    top_set = set(top_accounts_df['account'])
    flagged_tx_idx = tmp.index[(tmp[id_cols[0]].isin(top_set)) | (tmp[id_cols[1]].isin(top_set))]
    return top_accounts_df, flagged_tx_idx, account_mean

# -------------------------
# Full run function: ties everything together
# -------------------------
def run_chained_pipeline(path, target_col, sample_n=None, id_cols = ["Transaction_ID", "User_ID"]):
    print("Loading dataset...")
    df = load_df(path, sample_n=sample_n)
    if target_col not in df.columns:
        print("Target column not found.")
        return
    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
    X, y, enc = preprocess_for_models(df, target_col)
    print("Preprocessing done. Features:", X.shape)

    # supervised
    sup_prob, sup_model = supervised_scores(X, y)
    # unsupervised
    unsup_score, iso_model, scaler = unsupervised_scores(X, sup_prob, contamination='auto')
    # tune threshold
    best = tune_threshold_and_eval(y, sup_prob, unsup_score)

    # final preds
    final_pred = best['pred']   # from best entry
    # evaluate overall
    print("\n--- Final Evaluation (using tuned threshold) ---")
    print("Accuracy:", accuracy_score(y, final_pred))
    print(classification_report(y, final_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y, final_pred))

    # GNN-baseline
    top_accounts_df, flagged_tx_idx, account_mean = gnn_baseline_flagging(df, best['combined'], id_cols=id_cols, top_k_accounts=20)
    print("\nTop suspicious accounts (GNN-baseline):")
    print(top_accounts_df)

    combined = best['combined']
    flagged_idx = np.where(final_pred == 1)[0]
    flagged_df = df.iloc[flagged_idx][[*id_cols, 'Transaction_Amount', target_col]].assign(suspicion_score=combined[flagged_idx])
    print("\nNumber of transactions flagged (final):", len(flagged_df))
    flagged_df.to_csv("flagged_transactions.csv",  index=False)  # <<-- saves all flagged transactions
    print("All flagged transactions saved to 'flagged_transactions.csv'")

    return {
        'df': df,
        'X': X,
        'y': y,
        'encoders': enc,
        'scaler': scaler,
        'feature_columns': X.columns.tolist(),
        'sup_model': sup_model,
        'iso_model': iso_model,
        'sup_prob': sup_prob,
        'unsup_score': unsup_score,
        'best': best,
        'account_mean': account_mean,
        'flagged_idx': flagged_idx
    }

# -------------------------
# RUN (commented out - will be called from Flask app when needed)
# -------------------------
# res = run_chained_pipeline(dataset_path, target_col, sample_n=SAMPLE_N, id_cols=id_cols)