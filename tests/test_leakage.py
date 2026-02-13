import pytest
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config

def test_leakage_columns_are_dropped():
    """Test that processed data does not contain leakage columns defined in config."""
    config = load_config("configs/experiment.yaml")
    leakage_cols = config['preprocessing']['leakage_cols']
    
    # Load processed data
    # Assuming it exists. If not, this test should be skipped or fail gracefully.
    data_path = config['data']['processed_path']
    if not os.path.exists(data_path):
        pytest.skip(f"Data file {data_path} not found. Run preprocessing first.")
        
    df = pd.read_parquet(data_path)
    
    # Check for forbidden columns
    # Outcome IS allowed in the file (for training), but should be dropped from X
    # strictly forbidden: 'Allowed Amount', 'Paid Amount' etc.
    
    forbidden_cols = [c for c in leakage_cols if c != 'Outcome']
    
    present_forbidden = [c for c in forbidden_cols if c in df.columns]
    
    assert len(present_forbidden) == 0, f"Leakage columns found in processed data: {present_forbidden}"

def test_feature_manifest_exists():
    """Test that feature manifest matches dataset."""
    config = load_config("configs/experiment.yaml")
    data_path = config['data']['processed_path']
    manifest_path = os.path.join(os.path.dirname(data_path), 'feature_manifest.json')
    
    if not os.path.exists(data_path):
        pytest.skip("Data not found")
        
    assert os.path.exists(manifest_path)
    
    # Load manifest
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    df = pd.read_parquet(data_path)
    
    # Check that all manifest features are in df
    for feat in manifest['features']:
        assert feat in df.columns, f"Feature {feat} in manifest but not in dataframe"

if __name__ == "__main__":
    # Manually run if called as script
    test_leakage_columns_are_dropped()
    test_feature_manifest_exists()
    print("All leakage tests passed!")
