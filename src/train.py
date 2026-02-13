import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, set_seed, save_json, load_json

def train_models(data_path, split_path, output_dir, config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    split_info = load_json(split_path)
    
    # 1. Prepare Splits
    # We have train_patients, val_patients, test_patients
    # We need to map these to indices for PredefinedSplit (if using CV on Train+Val)
    # OR just train on Train, validate on Val for Tuning.
    
    # Let's use strict Train set for training, Val set for validation/early stopping.
    train_mask = df['Patient ID'].isin(split_info['train_patients'])
    val_mask = df['Patient ID'].isin(split_info['val_patients'])
    # Test set is reserved for FINAL evaluation, not touched here.
    
    X = df.drop(columns=['Outcome', 'Claim ID', 'Patient ID', 'Date of Service', 'Billed Amount']) 
    # Billed Amount is replaced by log, others are metadata.
    # Ensure we use feature_manifest if available to be safe?
    # manifest_path = os.path.join(os.path.dirname(data_path), 'feature_manifest.json')
    # if os.path.exists(manifest_path):
    #     features = load_json(manifest_path)['features']
    #     X = df[features]
    
    y = df['Outcome']
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    print(f"Training shapes: X_train={X_train.shape}, X_val={X_val.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Train Baseline (Logistic Regression)
    print("\nTraining Baseline (Logistic Regression)...")
    baseline = LogisticRegression(max_iter=1000, random_state=config['experiment']['seed'])
    baseline.fit(X_train, y_train)
    
    # Save Baseline
    joblib.dump(baseline, os.path.join(output_dir, "baseline_model.joblib"))
    print("Saved baseline model.")
    
    # 3. Train Strong Model (XGBoost) with Tuning
    print("\nTraining Strong Model (XGBoost) with Tuning...")
    
    # Define Parameter Space
    param_dist = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10)
    }
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=config['experiment']['seed'],
        n_jobs=-1
    )
    
    # Use PredefinedSplit for RandomizedSearchCV to use our fixed Validation set
    # -1 for training, 0 for validation
    split_index = np.full(X_train.shape[0] + X_val.shape[0], -1)
    split_index[X_train.shape[0]:] = 0 # Val indices
    
    ps = PredefinedSplit(test_fold=split_index)
    X_tune = pd.concat([X_train, X_val])
    y_tune = pd.concat([y_train, y_val])
    
    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=config['modeling']['tuning_trials'],
        scoring='roc_auc',
        cv=ps,
        verbose=1,
        random_state=config['experiment']['seed'],
        n_jobs=-1
    )
    
    search.fit(X_tune, y_tune)
    
    print(f"Best Params: {search.best_params_}")
    print(f"Best Val AUC: {search.best_score_:.4f}")
    
    best_model = search.best_estimator_
    
    # Save Strong Model
    joblib.dump(best_model, os.path.join(output_dir, "strong_model.joblib"))
    print("Saved strong model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/features.parquet')
    parser.add_argument('--split_path', default='splits/split_v1.json')
    parser.add_argument('--output_dir', default='models/exp_run')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    
    train_models(args.data_path, args.split_path, args.output_dir, args.config)

