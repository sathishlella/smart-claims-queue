import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate():
    n_records = 1000
    output_path = r'data/raw/claims_sample.csv'
    
    np.random.seed(1337)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
    
    data = {
        'Claim ID': [f"CLM{i:08d}" for i in range(n_records)],
        'Provider ID': [f"PRV{np.random.randint(1, 100):03d}" for _ in range(n_records)],
        'Patient ID': [f"PAT{np.random.randint(1, 1000):05d}" for _ in range(n_records)],
        'Date of Service': dates,
        'Billed Amount': np.round(np.random.lognormal(mean=5, sigma=1, size=n_records), 2),
        'Procedure Code': np.random.choice(['99213', '99214', '99203'], n_records),
        'Diagnosis Code': np.random.choice(['J00', 'I10', 'E11'], n_records),
        'Insurance Type': np.random.choice(['Medicare', 'Private'], n_records),
        'Claim Status': np.random.choice(['Paid', 'Denied'], n_records),
        'Outcome': np.random.choice(['Paid', 'Denied'], n_records, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    # Add leaky cols to ensure schema matches
    df['Allowed Amount'] = 0
    df['Paid Amount'] = 0
    df['Reason Code'] = ''
    df['Follow-up Required'] = False
    df['AR Status'] = 'Closed'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_records} records to {output_path}")

if __name__ == '__main__':
    generate()
