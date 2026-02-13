import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, set_seed

def calculate_priority(df, model_path, config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    weights = config['simulation']['weights']
    
    # Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model = joblib.load(model_path)
    
    # Prepare Features (Drop non-features)
    # Note: Ensure these drop columns match train.py exactly
    drop_cols = ['Outcome', 'Claim ID', 'Patient ID', 'Date of Service', 'Billed Amount'] 
    
    # We need to keep metadata for the final report, so we create X separately
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    X = X.loc[:, ~X.columns.duplicated()] # Dedup columns if any
    
    # Predict Risk Score
    # Handle if model expects specific columns (e.g. alignment)
    # If model has feature_names_in_, enforce it
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist() # Convert to list
        # Ensure it's a list of strings
        model_features = [str(f) for f in model_features]
        
        # Identify missing cols
        missing_cols = [c for c in model_features if c not in X.columns]
        if missing_cols:
            print(f"Warning: {len(missing_cols)} features missing in input, filling with 0.")
            for c in missing_cols:
                X[c] = 0
                
        # Reorder and drop extras using loc
        X = X.loc[:, model_features]
    else:
        # Fallback to just numeric if model doesn't have metadata (unlikely for sklearn/xgb)
        X = X.select_dtypes(include=[np.number, bool])
    
    X = X.reset_index(drop=True)
    print(f"Predicting with {model_path} on {X.shape}...")
    # Pass as numpy array to avoid pandas indexing issues in XGBoost/Sklearn
    df['risk_score'] = model.predict_proba(X.to_numpy())[:, 1]
    
    # 2. Calculate Components
    # Impact: Billed Amount (Log or Normalized)
    # We already have Billed Amount_log. Let's MinMax scale it to 0-1 for combination
    scaler = MinMaxScaler()
    if 'Billed Amount_log' in df.columns:
        df['impact_score'] = scaler.fit_transform(df[['Billed Amount_log']])
    else:
        df['impact_score'] = 0
    
    # Urgency: Days since DOS (Normalized)
    # Older claims might be MORE urgent (filing limit)
    # Assume 90 day limit. Close to 90 is urgent.
    # Current 'days_since_dos' is relative to 2024-01-01.
    # Let's just use raw days and scale.
    if 'days_since_dos' in df.columns:
        df['urgency_score'] = scaler.fit_transform(df[['days_since_dos']])
    else:
        df['urgency_score'] = 0
    
    # 3. Composite Score
    # P = w_r * Risk + w_i * Impact + w_u * Urgency
    w_r, w_i, w_u = weights['risk'], weights['impact'], weights['urgency']
    
    df['priority_score'] = (
        w_r * df['risk_score'] +
        w_i * df['impact_score'] +
        w_u * df['urgency_score']
    )
    
    # Rank
    df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    
    return df

def generate_ranking(data_path, model_path, output_path, config_path):
    df = pd.read_parquet(data_path)
    ranked_df = calculate_priority(df, model_path, config_path)
    
    # Save Report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export minimal columns for viewing
    export_cols = ['rank', 'priority_score', 'risk_score', 'impact_score', 'urgency_score', 
                   'Claim ID', 'Billed Amount', 'Date of Service', 'Outcome']
    
    # Outcome might not be in inference data, keep if exists
    final_cols = [c for c in export_cols if c in ranked_df.columns]
    
    ranked_df[final_cols].to_csv(output_path, index=False)
    print(f"Saved ranking to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/features.parquet')
    parser.add_argument('--model_path', default='models/exp_run/strong_model.joblib')
    parser.add_argument('--output_path', default='reports/priority_queue.csv')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    
    generate_ranking(args.data_path, args.model_path, args.output_path, args.config)
