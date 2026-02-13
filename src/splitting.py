import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
from sklearn.model_selection import GroupShuffleSplit

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, set_seed, save_json

def create_splits(data_path, output_path, config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Ensure Patient ID exists
    if 'Patient ID' not in df.columns:
        raise ValueError("Patient ID column missing. Cannot perform Group Split.")
    
    # Get unique patients
    patients = df['Patient ID'].unique()
    n_patients = len(patients)
    print(f"Total Patients: {n_patients}")
    
    # 1. Split Test (20%)
    test_size = config['data']['test_size']
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config['experiment']['seed'])
    train_val_idx, test_idx = next(gss_test.split(patients, groups=patients))
    
    train_val_patients = patients[train_val_idx]
    test_patients = patients[test_idx]
    
    # 2. Split Val (10% of total) from Train+Val
    # We want Val to be 10% of TOTAL.
    # Currently Train+Val is (1 - test_size) of total.
    # val_ratio = val_size / (1 - test_size)
    val_size_total = config['data']['val_size']
    val_ratio = val_size_total / (1 - test_size)
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=config['experiment']['seed'])
    train_idx, val_idx = next(gss_val.split(train_val_patients, groups=train_val_patients))
    
    train_patients = train_val_patients[train_idx]
    val_patients = train_val_patients[val_idx]
    
    print(f"Split Counts (Patients):")
    print(f"  Train: {len(train_patients)} ({len(train_patients)/n_patients:.1%})")
    print(f"  Val:   {len(val_patients)} ({len(val_patients)/n_patients:.1%})")
    print(f"  Test:  {len(test_patients)} ({len(test_patients)/n_patients:.1%})")
    
    # Verify no overlap
    assert len(set(train_patients) & set(test_patients)) == 0
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(val_patients) & set(test_patients)) == 0
    
    split_info = {
        'train_patients': train_patients.tolist(),
        'val_patients': val_patients.tolist(),
        'test_patients': test_patients.tolist()
    }
    
    save_json(split_info, output_path)
    print(f"Saved split info to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/features.parquet')
    parser.add_argument('--output_path', default='splits/split_v1.json')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    
    create_splits(args.data_path, args.output_path, args.config)
