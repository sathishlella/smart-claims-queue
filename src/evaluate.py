import pandas as pd
import numpy as np
import argparse
import joblib
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import load_config, set_seed, save_json, load_json

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip samples with only one class
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    mean_score = np.mean(sorted_scores)
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return mean_score, ci_lower, ci_upper

def evaluate_models(data_path, split_path, model_dir, output_dir, config_path="configs/experiment.yaml"):
    config = load_config(config_path)
    set_seed(config['experiment']['seed'])
    
    print("Loading data and splits...")
    df = pd.read_parquet(data_path)
    split_info = load_json(split_path)
    
    test_mask = df['Patient ID'].isin(split_info['test_patients'])
    
    # Prepare X_test, y_test
    X = df.drop(columns=['Outcome', 'Claim ID', 'Patient ID', 'Date of Service', 'Billed Amount'])
    y = df['Outcome']
    
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"Test Set: {len(X_test)} samples, {y_test.sum()} positives ({y_test.mean():.2%} prevalence)")
    
    results = {}
    
    models = {
        'baseline': os.path.join(model_dir, 'baseline_model.joblib'),
        'strong': os.path.join(model_dir, 'strong_model.joblib')
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Model {name} not found at {path}, skipping.")
            continue
            
        print(f"Evaluating {name}...")
        model = joblib.load(path)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auroc, auroc_l, auroc_u = bootstrap_ci(y_test, y_prob, roc_auc_score)
        auprc, auprc_l, auprc_u = bootstrap_ci(y_test, y_prob, average_precision_score)
        brier = brier_score_loss(y_test, y_prob) # No CI for Brier usually needed, but good to have
        
        print(f"  AUROC: {auroc:.4f} [{auroc_l:.4f}, {auroc_u:.4f}]")
        print(f"  AUPRC: {auprc:.4f} [{auprc_l:.4f}, {auprc_u:.4f}]")
        print(f"  Brier: {brier:.4f}")
        
        results[name] = {
            'auroc': {'mean': auroc, 'ci_lower': auroc_l, 'ci_upper': auroc_u},
            'auprc': {'mean': auprc, 'ci_lower': auprc_l, 'ci_upper': auprc_u},
            'brier': brier
        }
        
        # Calibration Plot
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{name} (Brier={brier:.3f})')
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'calibration_plot.png'))
    print("Saved calibration plot.")
    
    save_json(results, os.path.join(output_dir, 'evaluation_results.json'))
    print(f"Saved results to {os.path.join(output_dir, 'evaluation_results.json')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/features.parquet')
    parser.add_argument('--split_path', default='splits/split_v1.json')
    parser.add_argument('--model_dir', default='models/exp_run')
    parser.add_argument('--output_dir', default='reports/exp_run')
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    
    evaluate_models(args.data_path, args.split_path, args.model_dir, args.output_dir, args.config)
