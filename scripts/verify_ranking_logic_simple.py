import pandas as pd
import os

def verify_ranking():
    file_path = "reports/priority_queue.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Basic Stats
    print(f"Total Claims: {len(df)}")
    
    # 2. Tier Counts
    high = df[df['priority_score'] >= 0.6]
    med = df[(df['priority_score'] >= 0.3) & (df['priority_score'] < 0.6)]
    low = df[df['priority_score'] < 0.3]
    
    print("\n--- Risk Tiers (Full Dataset) ---")
    print(f"High Risk (>=0.6): {len(high)} ({len(high)/len(df):.1%})")
    print(f"Med Risk (0.3-0.6): {len(med)} ({len(med)/len(df):.1%})")
    print(f"Low Risk (<0.3): {len(low)} ({len(low)/len(df):.1%})")
    
    # 3. Where do Low Risk claims start?
    # Ensure sorted
    df_sorted = df.sort_values(by='priority_score', ascending=False).reset_index(drop=True)
    
    # Find first index where score < 0.3
    low_risk_indices = df_sorted.index[df_sorted['priority_score'] < 0.3]
    
    if len(low_risk_indices) > 0:
        first_low_rank = low_risk_indices[0] + 1
        print(f"\n--- Ranking Cutoffs ---")
        print(f"The first 'Low Priority' claim appears at Rank: {first_low_rank}")
        print(f"This means you must work through {first_low_rank - 1} High/Medium claims before seeing a Low one.")
        
        # Check specific capacities
        capacities = [10, 46, 50, 100]
        for cap in capacities:
            worklist_size = cap * 5
            cutoff_score = df_sorted.iloc[worklist_size-1]['priority_score']
            status = "Only High/Medium claims visible." if cutoff_score >= 0.4 else "Includes Low claims."
            print(f"At Capacity {cap}/day (Worklist {worklist_size}): Low Score Cutoff = {cutoff_score:.4f} -> {status}")

if __name__ == "__main__":
    verify_ranking()
