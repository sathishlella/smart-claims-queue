import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def verify_ranking():
    file_path = "reports/priority_queue.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Basic Stats
    print(f"Total Claims: {len(df)}")
    print(df['priority_score'].describe())
    
    # 2. Tier Counts
    high = df[df['priority_score'] >= 0.7]
    med = df[(df['priority_score'] >= 0.4) & (df['priority_score'] < 0.7)]
    low = df[df['priority_score'] < 0.4]
    
    print("\n--- Risk Tiers (Full Dataset) ---")
    print(f"High Risk (>=0.7): {len(high)} ({len(high)/len(df):.1%})")
    print(f"Med Risk (0.4-0.7): {len(med)} ({len(med)/len(df):.1%})")
    print(f"Low Risk (<0.4): {len(low)} ({len(low)/len(df):.1%})")
    
    # 3. Where do Low Risk claims start?
    # Ensure sorted
    df_sorted = df.sort_values(by='priority_score', ascending=False).reset_index(drop=True)
    
    # Find first index where score < 0.4
    low_risk_indices = df_sorted.index[df_sorted['priority_score'] < 0.4]
    
    if len(low_risk_indices) > 0:
        first_low_rank = low_risk_indices[0]
        print(f"\n--- Ranking Cutoffs ---")
        print(f"The first 'Low Priority' claim appears at Rank: {first_low_rank + 1}")
        print(f"This means you must work through {first_low_rank} High/Medium claims before seeing a Low one.")
        
        # Check specific capacities
        capacities = [10, 50, 100]
        for cap in capacities:
            worklist_size = cap * 5
            cutoff_score = df_sorted.iloc[worklist_size-1]['priority_score']
            print(f"At Capacity {cap}/day (Worklist {worklist_size}): Low Score Cutoff = {cutoff_score:.4f}")
            if cutoff_score >= 0.4:
                print(f"  -> result: Only High/Medium claims visible.")
            else:
                print(f"  -> result: Includes Low claims.")
                
    # 4. Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['priority_score'], bins=50, kde=True)
    plt.axvline(0.4, color='g', linestyle='--', label='Low/Med Cutoff (0.4)')
    plt.axvline(0.7, color='r', linestyle='--', label='Med/High Cutoff (0.7)')
    plt.title('Distribution of Priority Scores')
    plt.xlabel('Priority Score')
    plt.legend()
    plt.savefig('reports/priority_score_dist.png')
    print("\nDistribution plot saved to reports/priority_score_dist.png")

if __name__ == "__main__":
    verify_ranking()
