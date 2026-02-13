import pandas as pd
import numpy as np
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime, timedelta

def generate_synthetic_data(n_records, output_path, seed=1337):
    np.random.seed(seed)
    
    print(f"Generating {n_records} synthetic records...")
    
    # Generate dates (2023-2024 range)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_records)]
    
    # Generate IDs
    claim_ids = [f"CLM{i:08d}" for i in range(n_records)]
    # Providers: 1000 unique providers (skewed distribution)
    provider_pool = [f"PRV{i:05d}" for i in range(1000)]
    provider_ids = np.random.choice(provider_pool, n_records, p=np.random.dirichlet(np.ones(1000)*0.5)) 
    
    # Patients: 10,000 unique patients
    patient_ids = [f"PAT{np.random.randint(1, 10001):05d}" for _ in range(n_records)]
    
    # Categorical fields
    insurance_types = ['Medicare', 'Medicaid', 'BlueCross', 'Aetna', 'UnitedHealth', 'Cigna', 'Self-Pay']
    proc_codes = ['99213', '99214', '99203', '99204', '90837', '90834', '99285', '99291']
    diag_codes = ['J01.90', 'I10', 'E11.9', 'F32.9', 'M54.5', 'R07.9', 'Z00.00']
    
    # Logic for Denial
    # We want some signal. 
    # E.g. Certain Insurance + Proc Combo = Higher Denial
    
    df = pd.DataFrame({
        'Claim ID': claim_ids,
        'Provider ID': provider_ids,
        'Patient ID': patient_ids,
        'Date of Service': dates,
        'Billed Amount': np.round(np.random.lognormal(mean=5.5, sigma=1.2, size=n_records), 2),
        'Procedure Code': np.random.choice(proc_codes, n_records),
        'Diagnosis Code': np.random.choice(diag_codes, n_records),
        'Insurance Type': np.random.choice(insurance_types, n_records, p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]),
    })
    
    # Generate Status/Outcome with some pattern
    def determine_status(row):
        prob_denial = 0.15 # Base rate
        
        # Risk factors
        if row['Insurance Type'] == 'Medicaid': prob_denial += 0.10
        if row['Procedure Code'] == '99285': prob_denial += 0.20 # Emergency high level
        if row['Billed Amount'] > 5000: prob_denial += 0.15
        
        # Random chance
        if np.random.random() < prob_denial:
            return 'Denied'
        else:
            return 'Paid'

    df['Claim Status'] = df.apply(determine_status, axis=1)
    df['Outcome'] = df['Claim Status'].apply(lambda x: 'Denied' if x == 'Denied' else 'Paid')
    
    # Leaky columns (for realism, but dropped later)
    df['Allowed Amount'] = df.apply(lambda x: x['Billed Amount'] * 0.7 if x['Outcome'] == 'Paid' else 0, axis=1)
    df['Paid Amount'] = df.apply(lambda x: x['Allowed Amount'] * 0.9 if x['Outcome'] == 'Paid' else 0, axis=1)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='generate', choices=['generate', 'load'])
    parser.add_argument('--rows', type=int, default=100000, help="Number of rows")
    parser.add_argument('--output', type=str, default='data/raw/synthera835_claims.csv')
    parser.add_argument('--seed', type=int, default=1337)
    
    args = parser.parse_args()
    
    if args.mode == 'generate':
        generate_synthetic_data(args.rows, args.output, args.seed)

if __name__ == '__main__':
    main()
