import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_simulation(json_path, output_dir):
    print(f"Loading simulation data from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    curves = data['curves']
    stats = data['stats']
    
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    # JSON stores lists: 'days', 'savings'
    for name, curve_data in curves.items():
        plt.plot(curve_data['days'], curve_data['savings'], label=name, linewidth=2)
        
    plt.xlabel('Days of Work (Capacity=50)')
    plt.ylabel('Cumulative Revenue Prevented ($)')
    plt.title('Workflow Efficiency: AI Prioritization vs FIFO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add stats annotation
    if 'day_10_lift' in stats:
        lift = stats['day_10_lift']
        plt.annotate(f"Day 10 Lift: {lift:.1f}x", 
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round", fc="w"))
    
    plot_path = os.path.join(output_dir, 'simulation.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved simulation plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='reports/exp_scientific/simulation_data.json')
    parser.add_argument('--output_dir', default='reports/exp_scientific')
    args = parser.parse_args()
    
    plot_simulation(args.json_path, args.output_dir)
