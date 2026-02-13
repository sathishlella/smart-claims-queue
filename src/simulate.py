import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import traceback
from sklearn.preprocessing import MinMaxScaler
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, load_json, save_json

def run_simulation_robust(data_path, split_path, model_path, output_dir, config_path="configs/experiment.yaml"):
    # Load Config from arguments
    print("DEBUG: Starting Simulation (Compute Only)...")
    
    config = load_config(config_path)
    capacity = config['simulation']['daily_capacity']
    weights = config['simulation']['weights']
    
    print("DEBUG: Loading data...")
    df = pd.read_parquet(data_path)
    
    print("DEBUG: Loading splits...")
    with open(split_path, 'r') as f:
        split_info = json.load(f)
    test_patients = set(split_info['test_patients'])
    
    print("DEBUG: Filtering test set...")
    test_mask = df['Patient ID'].isin(test_patients)
    df_test = df[test_mask].copy()
    print(f"DEBUG: Test set shape: {df_test.shape}")
    
    print("DEBUG: Loading model...")
    model = joblib.load(model_path)
    
    drop_cols = ['Outcome', 'Claim ID', 'Patient ID', 'Date of Service', 'Billed Amount'] 
    
    print("DEBUG: Cleaning DF...")
    df_test = df_test.reset_index(drop=True)
    X = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')
    X = X.loc[:, ~X.columns.duplicated()]
    
    print("DEBUG: Checking feature_names_in_...")
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist()
        model_features = [str(f) for f in model_features]
        
        missing = [c for c in model_features if c not in X.columns]
        for c in missing:
            X[c] = 0
            
        print("DEBUG: Reordering columns manually...")
        X_new = pd.DataFrame(index=X.index)
        for col in model_features:
            X_new[col] = X[col]
        X = X_new
        print(f"DEBUG: X reordered. Shape: {X.shape}")
    else:
        X = X.select_dtypes(include=[np.number, bool])
        
    print("DEBUG: Converting to numpy...")
    X_np = X.to_numpy()
    print(f"DEBUG: X_np shape: {X_np.shape}, dtype: {X_np.dtype}")
    
    print("DEBUG: Predicting...")
    try:
        # Use simple assignment
        probs = model.predict_proba(X_np)[:, 1]
        df_test['risk_score'] = probs
        print("DEBUG: Prediction success!")
    except Exception as e:
        print("DEBUG: Prediction failed!")
        traceback.print_exc()
        sys.exit(1)

    # Scoring Components
    scaler = MinMaxScaler()
    
    if 'Billed Amount_log' in df_test.columns:
        val = df_test['Billed Amount_log'].to_numpy().reshape(-1, 1)
        df_test['impact_score'] = scaler.fit_transform(val)
    else:
        df_test['impact_score'] = 0
        
    if 'days_since_dos' in df_test.columns:
        val = df_test['days_since_dos'].to_numpy().reshape(-1, 1)
        df_test['urgency_score'] = scaler.fit_transform(val)
    else:
        df_test['urgency_score'] = 0
        
    w_r, w_i, w_u = weights['risk'], weights['impact'], weights['urgency']
    
    df_test['priority_score'] = (
        w_r * df_test['risk_score'] +
        w_i * df_test['impact_score'] +
        w_u * df_test['urgency_score']
    )
    
    # Simulation Logic (Compute Only)
    print("DEBUG: Starting Simulation Computation...")
    
    strategies = {
        'FIFO': df_test.sort_values('Date of Service', ascending=True),
        'AI_Priority': df_test.sort_values('priority_score', ascending=False)
    }
    
    os.makedirs(output_dir, exist_ok=True)
    sim_data = {}
    
    for name, df_sorted in strategies.items():
        df_sorted = df_sorted.copy().reset_index(drop=True)
        
        # Cumulative Savings
        df_sorted['potential_savings'] = df_sorted['Outcome'] * df_sorted['Billed Amount']
        df_sorted['cumulative_savings'] = df_sorted['potential_savings'].cumsum()
        
        # Days
        df_sorted['day'] = (df_sorted.index // capacity) + 1
        
        daily_stats = df_sorted.groupby('day')['cumulative_savings'].max()
        
        # Convert keys/values to list for JSON serialization
        sim_data[name] = {
            'days': daily_stats.index.tolist(),
            'savings': daily_stats.values.tolist()
        }
        
    # Lift Stats
    # Get day 10 savings (search in list)
    try:
        idx_10 = sim_data['FIFO']['days'].index(10)
        day_10_fifo = sim_data['FIFO']['savings'][idx_10]
        
        idx_10_ai = sim_data['AI_Priority']['days'].index(10)
        day_10_ai = sim_data['AI_Priority']['savings'][idx_10_ai]
        
        lift = day_10_ai / day_10_fifo if day_10_fifo > 0 else 0
        print(f"Day 10 Savings: FIFO=${day_10_fifo:,.0f}, AI=${day_10_ai:,.0f} (Lift={lift:.2f}x)")
        
        sim_stats = {
            'day_10_lift': lift,
            'ai_savings_day_10': float(day_10_ai),
            'fifo_savings_day_10': float(day_10_fifo)
        }
    except ValueError:
        sim_stats = {'error': 'Day 10 not found'}

    # Save Data
    json_path = os.path.join(output_dir, 'simulation_data.json')
    save_json({'curves': sim_data, 'stats': sim_stats}, json_path)
    print(f"Saved simulation data to {json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/features.parquet')
    parser.add_argument('--split_path', default='splits/split_v1.json')
    parser.add_argument('--model_path', default='models/exp_scientific/strong_model.joblib')
    parser.add_argument('--output_dir', default='reports/exp_scientific')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    
    run_simulation_robust(args.data_path, args.split_path, args.model_path, args.output_dir, args.config)
