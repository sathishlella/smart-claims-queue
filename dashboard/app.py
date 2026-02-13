import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import joblib
import sys

# Ensure project root is in path for 'src' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def display_risk_metrics(df):
    # Calculate Risk Tiers (Adjusted Thresholds)
    high_risk = df[df['priority_score'] >= 0.6].shape[0]
    med_risk = df[(df['priority_score'] >= 0.3) & (df['priority_score'] < 0.6)].shape[0]
    low_risk = df[df['priority_score'] < 0.3].shape[0]
    
    st.subheader("Queue Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("🔥 High Priority", high_risk, help="Score >= 0.6")
    m2.metric("⚠️ Medium Priority", med_risk, help="0.3 <= Score < 0.6")
    m3.metric("✅ Low Priority", low_risk, help="Score < 0.3")
    
st.set_page_config(page_title="Claims Denial Prediction", layout="wide")

st.title("🏥 Claims Denial Prediction & Ranking Dashboard")

# sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Data View", "Model Evaluation", "Priority Queue"])

if page == "Home":
    st.markdown("""
    ## Overview
    This dashboard provides insights into medical claim denials using a machine learning pipeline.
    
    ### Key Features
    - **Upload Data**: Analyze your own claims file
    - **Data Ingestion**: View synthetic claims data
    - **Modeling**: Compare Baseline vs Strong models
    - **Evaluation**: AUROC, Calibration, and Performance Metrics
    - **Ranking**: Prioritize claims for follow-up
    """)
    
    if os.path.exists("reports/exp_scientific/simulation_stats.json"):
        with open("reports/exp_scientific/simulation_stats.json", "r") as f:
            stats = json.load(f)
        st.subheader("Latest Simulation Performance")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ROI Lift (Day 10)", f"{stats.get('day_10_lift', 0):.2f}x")
        with c2:
            st.metric("Revenue Recovered (AI)", f"${stats.get('ai_savings_day_10', 0):,.0f}")

if page == "Upload Data":
    st.header("Upload Claims Data")
    st.markdown("Upload a CSV file containing claims data to generate a priority queue.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Save file
        upload_dir = "data/raw/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, "uploaded_claims.csv")
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"File saved to {file_path}")
        
        # Preview
        df = pd.read_csv(file_path)
        st.write("Preview:", df.head())
        
        if st.button("Process & Rank"):
            with st.spinner("Running pipeline..."):
                try:
                    import subprocess
                    import sys
                    
                    # 1. Preprocessing
                    st.text("Preprocessing...")
                    cmd_preprocess = [
                        sys.executable, "src/preprocessing.py",
                        "--input_path", file_path,
                        "--output_path", "data/processed/uploaded_features.parquet"
                    ]
                    result = subprocess.run(cmd_preprocess, capture_output=True, text=True)
                    if result.returncode != 0:
                        st.error(f"Preprocessing failed:\n{result.stderr}")
                        st.stop()
                    
                    # 2. Ranking (Using ranking_v2)
                    st.text("Ranking...")
                    cmd_rank = [
                        sys.executable, "src/ranking_v2.py",
                        "--data_path", "data/processed/uploaded_features.parquet",
                        "--model_path", "models/exp_scientific/strong_model.joblib",
                        "--output_path", "reports/priority_queue.csv",
                        "--config", "configs/experiment.yaml"
                    ]
                    result_rank = subprocess.run(cmd_rank, capture_output=True, text=True)
                    if result_rank.returncode != 0:
                         st.error(f"Ranking failed:\n{result_rank.stderr}")
                         st.stop()
                    
                    st.success("Processing Complete! Results below:")
                    
                    # Display metrics immediately using the generated file
                    if os.path.exists("reports/priority_queue.csv"):
                        df_res = pd.read_csv("reports/priority_queue.csv")
                        display_risk_metrics(df_res)
                        st.dataframe(df_res.head())
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    if os.path.exists("reports/priority_queue.csv"):
        st.write("Recent Results Available:")
        df_recent = pd.read_csv("reports/priority_queue.csv")
        display_risk_metrics(df_recent)

if page == "Data View":
    st.header("Data Inspector")
    
    # Check for uploaded file first
    uploaded_path = "data/raw/uploads/uploaded_claims.csv"
    sample_path = "data/raw/claims_sample.csv"
    
    if os.path.exists(uploaded_path):
        df_raw = pd.read_csv(uploaded_path)
        st.success(f"Loaded Uploaded Data: {uploaded_path}")
        st.write(f"Shape: {df_raw.shape}")
        st.dataframe(df_raw.head(100))
    elif os.path.exists(sample_path):
        df_raw = pd.read_csv(sample_path)
        st.info(f"Loaded Sample Data: {sample_path}")
        st.write(f"Shape: {df_raw.shape}")
        st.dataframe(df_raw.head(100))
    else:
        st.warning("No data found. Please upload a file in the 'Upload Data' section.")

if page == "Model Evaluation":
    st.header("Model Performance")
    
    if os.path.exists("reports/exp_scientific/evaluation_results.json"):
        with open("reports/exp_scientific/evaluation_results.json", "r") as f:
            metrics = json.load(f)
        st.json(metrics)
        
    if os.path.exists("reports/exp_scientific/calibration_plot.png"):
        st.image("reports/exp_scientific/calibration_plot.png", caption="Calibration Curve")
        
    if os.path.exists("reports/exp_scientific/simulation.png"):
        st.image("reports/exp_scientific/simulation.png", caption="Workflow Simulation")

if page == "Priority Queue":
    st.header("Worklist Prioritization")
    
    capacity = st.slider("Capacity (Claims/Day)", 10, 100, 50)
    
    if st.button("Generate Queue (Test Data)"):
        try:
            from src.ranking_v2 import generate_ranking
            generate_ranking(
                'data/processed/features.parquet',
                'models/exp_scientific/strong_model.joblib',
                'reports/priority_queue.csv',
                'configs/experiment.yaml'
            )
            st.success("Queue generated from Test Data!")
        except Exception as e:
            st.error(f"Error generating queue: {e}")

    if os.path.exists("reports/priority_queue.csv"):
        df_q = pd.read_csv("reports/priority_queue.csv")
        
        # Filter based on capacity (Top 5 Days)
        top_n = capacity * 5
        
        # Ensure it is sorted by priority score descending
        df_q = df_q.sort_values(by="priority_score", ascending=False)
        
        df_view = df_q.head(top_n)
        
        
        st.markdown("### Worklist Overview (Top N)")
        display_risk_metrics(df_view)
        st.caption(f"Showing top {top_n} claims (5-day worklist at {capacity} claims/day)")
        
        with st.expander("📊 View Full Backlog Statistics (All Claims)"):
            st.markdown("These are the counts for the **entire dataset**, including claims not in the current worklist.")
            display_risk_metrics(df_q)
        
        
        # Filter top N based on capacity * days? Or just show all?
        # Let's show all but highlight top capacity
        
        st.subheader("Ranked Claims")
        st.dataframe(df_q.head(capacity * 5)) # Show top 5 days worth
        
        st.download_button(
            "Download Full List CSV",
            df_q.to_csv(index=False),
            "priority_queue.csv",
            "text/csv"
        )
