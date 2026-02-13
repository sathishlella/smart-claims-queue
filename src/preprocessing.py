import pandas as pd
import numpy as np
import argparse
import hashlib
import os
import sys

# Add src to path if running as script
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, set_seed, save_json

def list_features(df, exclude_cols):
    return [c for c in df.columns if c not in exclude_cols]

def preprocess_data(input_path, output_path, config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 0. Schema Mapping (Synthera835 -> Internal)
    # Remove index column if present
    df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')

    # 0. Schema Mapping (Synthera835 -> Internal)
    # Define aliases (target: [possible_sources])
    aliases = {
        'Claim ID': ['claim_id', 'Claim ID'],
        'Patient ID': ['patient_id', 'Patient ID'],
        'Provider ID': ['provider_npi', 'provider_id', 'Provider ID'],
        'Insurance Type': ['payer_id', 'insurance_type', 'Insurance Type'],
        'Date of Service': ['date_of_service', 'Date of Service'],
        'Procedure Code': ['procedure_codes', 'procedure_code', 'Procedure Code'],
        'Diagnosis Code': ['diagnosis_codes', 'diagnosis_code', 'Diagnosis Code'],
        'Billed Amount': ['total_charge', 'billed_amount', 'Billed Amount'],
        'Paid Amount': ['total_paid', 'paid_amount', 'Paid Amount'],
        'Claim Status': ['claim_status', 'Claim Status'],
        'Reason Code': ['denial_category', 'reason_code', 'Reason Code']
    }
    
    for target, sources in aliases.items():
        for source in sources:
            if source in df.columns:
                df = df.rename(columns={source: target})
                break

    # 1. Target Definition
    if 'Claim Status' in df.columns:
        df['Claim Status'] = df['Claim Status'].astype(str).str.lower()
        # 'denied' is 1, everything else (paid, partial) is 0
        df['Outcome'] = (df['Claim Status'] == 'denied').astype(int)
    elif 'Outcome' not in df.columns:
         print("Warning: Neither 'Claim Status' nor 'Outcome' found. Inference mode.")
    
    # 2. Strict Leakage Removal
    leakage_cols = config['preprocessing']['leakage_cols']
    # Keep Outcome if it exists (for splitting/training), but remember to drop it from X later
    # The config says Outcome is in leakage_cols, but we need it for y.
    # We will drop columns that are NOT the target but are leakage.
    
    cols_to_drop_now = [c for c in leakage_cols if c in df.columns and c != 'Outcome']
    print(f"Dropping leakage columns: {cols_to_drop_now}")
    df = df.drop(columns=cols_to_drop_now)
    
    # 3. Feature Engineering
    # Date Handling
    if 'Date of Service' in df.columns:
        df['Date of Service'] = pd.to_datetime(df['Date of Service'], errors='coerce').fillna(pd.Timestamp('2024-01-01'))
    else:
        df['Date of Service'] = pd.Timestamp('2024-01-01')
        
    ref_date = pd.Timestamp('2024-01-01')
    df['days_since_dos'] = (ref_date - df['Date of Service']).dt.days
    df['month'] = df['Date of Service'].dt.month
    df['dow'] = df['Date of Service'].dt.dayofweek
    
    # 4. Deterministic Hashing (Ablation B: Hashing)
    # Using MD5 for specific IDs
    
    def hash_col(series, limit):
        return series.astype(str).apply(lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16) % limit)

    buckets = config['preprocessing']['hash_buckets']
    
    if 'Provider ID' in df.columns:
        df['Provider ID'] = hash_col(df['Provider ID'], buckets['provider'])
    else:
        df['Provider ID'] = 0
        
    if 'Procedure Code' in df.columns:
        df['Procedure Code'] = hash_col(df['Procedure Code'], buckets['procedure'])
    else:
        df['Procedure Code'] = 0
        
    if 'Insurance Type' in df.columns:
        df['Insurance Type'] = hash_col(df['Insurance Type'], buckets['insurance'])
    else:
         df['Insurance Type'] = 0
         
    # Diagnosis Code: Keep as LabelEncoded (or hash if huge). 
    # For now, let's hash it too for consistency with the "Hashing" ablation
    if 'Diagnosis Code' in df.columns:
         # Use a larger bucket for diagnosis
         df['Diagnosis Code'] = hash_col(df['Diagnosis Code'], 5000)
    else:
        df['Diagnosis Code'] = 0

    # Billed Amount
    if 'Billed Amount' in df.columns:
        df['Billed Amount'] = pd.to_numeric(df['Billed Amount'], errors='coerce').fillna(0)
        df['Billed Amount_log'] = np.log1p(df['Billed Amount'])
    else:
        df['Billed Amount_log'] = 0.0

    # 5. Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    # Save Feature Manifest
    # Features are everything EXCEPT Outcome, Claim ID, Patient ID, Date of Service
    non_feature_cols = ['Outcome', 'Claim ID', 'Patient ID', 'Date of Service', 'Billed Amount'] # Billed Amount is replaced by log
    features = [c for c in df.columns if c not in non_feature_cols]
    
    manifest_path = os.path.join(os.path.dirname(output_path), 'feature_manifest.json')
    save_json({'features': features}, manifest_path)
    print(f"Saved feature manifest to {manifest_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='data/raw/synthera835_claims.csv')
    parser.add_argument('--output_path', default='data/processed/features.parquet')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    
    preprocess_data(args.input_path, args.output_path, args.config)
