import argparse
import json
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema
    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False

def validate_schema(data_path, output_report):
    df = pd.read_csv(data_path)
    
    report = {"status": "skipped", "errors": []}
    
    if HAS_PANDERA:
        # Define Schema
        schema = DataFrameSchema({
            "Claim ID": Column(str, Check(lambda s: s.str.startswith("CLM")), unique=True, coerce=True),
            "Provider ID": Column(str, Check(lambda s: s.str.startswith("PRV")), coerce=True),
            "Patient ID": Column(str, Check(lambda s: s.str.startswith("PAT")), coerce=True),
            "Date of Service": Column(pd.Timestamp, coerce=True),
            "Billed Amount": Column(float, Check.greater_than_or_equal_to(0)),
            # "Procedure Code": Column(str), # Commented out as type might vary
            # "Diagnosis Code": Column(str), 
            "Insurance Type": Column(str, Check.isin(['Medicare', 'Medicaid', 'Private', 'Self-Pay'])),
            "Claim Status": Column(str),
            "Outcome": Column(str, Check.isin(['Paid', 'Denied'])),
        })
        
        try:
            schema.validate(df, lazy=True)
            report = {"status": "success", "errors": []}
            print("Validation Passed")
        except pa.errors.SchemaErrors as err:
            report = {"status": "failure", "errors": err.failure_cases.to_dict(orient='records')}
            print("Validation Failed")
            # print(err.failure_cases)
    else:
        print("Pandera not installed. Skipping strict schema validation.")
        # Minimal checks
        required_cols = ["Claim ID", "Outcome", "Billed Amount"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            report = {"status": "failure", "errors": [f"Missing columns: {missing}"]}
            print(f"Validation Failed: Missing {missing}")
        else:
            report = {"status": "success", "errors": []}
            print("Basic Validation Passed")

    with open(output_report, 'w') as f:
        json.dump(report, f, indent=2, default=str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('output_report')
    args = parser.parse_args()
    
    validate_schema(args.data_path, args.output_report)
