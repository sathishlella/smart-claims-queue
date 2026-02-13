import pandas as pd
import json
import os

def generate_manuscript_artifacts():
    # Load metrics
    with open('reports/metrics.json', 'r') as f:
        metrics = json.load(f)
        
    # Table 2: Model Performance
    table2 = pd.DataFrame(metrics).T
    table2.to_csv('reports/table2_model_performance.csv')
    
    # Table 1: Cohort Summary
    df = pd.read_csv('data/raw/claims_sample.csv')
    summary = {
        'Total Claims': len(df),
        'Unique Patients': df['Patient ID'].nunique(),
        'Unique Providers': df['Provider ID'].nunique(),
        'Denial Rate': df['Outcome'].value_counts(normalize=True).get('Denied', 0.0),
        'Mean Billed Amount': df['Billed Amount'].mean(),
        'Median Billed Amount': df['Billed Amount'].median()
    }
    table1 = pd.DataFrame([summary])
    table1.to_csv('reports/table1_cohort_summary.csv', index=False)
    
    # Results JSON
    results = {
        'experiment_id': 'EXP_20260212_RUN001',
        'cohort_size': len(df),
        'denial_rate': summary['Denial Rate'],
        'best_model': 'strong' if metrics['strong']['AUROC'] > metrics['baseline']['AUROC'] else 'baseline',
        'best_auroc': max(metrics['strong']['AUROC'], metrics['baseline']['AUROC']),
        'improvement_over_baseline': metrics['strong']['AUROC'] - metrics['baseline']['AUROC']
    }
    with open('reports/results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Generated manuscript artifacts in reports/")

if __name__ == '__main__':
    generate_manuscript_artifacts()
