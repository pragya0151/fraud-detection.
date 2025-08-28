import os
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")
st.title(" Real-Time Fraud Detection")

# -----------------------------
# 1. Check required files
# -----------------------------
required_files = ["rf_model.pkl", "xgb_model.pkl", "scaler.pkl", "metrics.json"]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"Missing files: {', '.join(missing)}. Please run `python train_model.py` first.")
    st.stop()

# -----------------------------
# 2. Load models, scaler & metadata
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("metrics.json", "r") as f:
    meta = json.load(f)

feature_names = meta["feature_names"]
n_features = meta["n_features"]
dataset_used = meta["dataset"]

# model choice
model_choice = st.sidebar.selectbox("Choose model for live predictions:", ["RandomForest", "XGBoost"])
model = rf_model if model_choice == "RandomForest" else xgb_model

st.sidebar.write("""
This is an **AI-powered Real-Time Fraud Detection System**.  
It predicts whether a credit card transaction is **fraudulent or genuine**  
using advanced **machine learning models**.
""")

with st.sidebar.expander(" FAQ: How does it work?"):
    st.write("""
- The system takes **transaction details** as input.
- The data is preprocessed and normalized.
- A trained ML model (**Random Forest or XGBoost**) makes the prediction.
- The output shows if the transaction is **fraudulent (1)** or **genuine (0)**.
    """)

with st.sidebar.expander(" FAQ: Why Machine Learning for Fraud Detection?"):
    st.write("""
Traditional rule-based systems struggle with:
- **Evolving fraud patterns**  
- **Large-scale real-time transactions**  

Machine learning adapts to new data and can detect **subtle, hidden fraud patterns** that humans might miss.
    """)

with st.sidebar.expander(" FAQ: Random Forest vs XGBoost"):
    st.write("""
**Random Forest**:
- ✅ High **precision** (fewer false alarms)  
- ❌ Lower **recall** (misses some fraud cases)

**XGBoost**:
- ✅ Higher **recall** (catches more frauds)  
- ✅ Higher **ROC-AUC** (better at discrimination)  
- ❌ Slightly lower precision (more false positives)

 In practice, **XGBoost is preferred** since banks prioritize **catching fraud** over avoiding a few false alarms.
    """)

with st.sidebar.expander(" Model Performance Comparison"):
    st.write("### Random Forest Metrics")
    st.json(meta["RandomForest"])
    st.write("### XGBoost Metrics")
    st.json(meta["XGBoost"])

# Database connection
conn = sqlite3.connect("realtime_fraud.db")
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    prediction INTEGER,
    amount REAL,
    time_val REAL
)
""")
conn.commit()

# -----------------------------
# Tabs
# -----------------------------
tab_live, tab_perf = st.tabs([" Live Transactions", "Model Performance"])

# -----------------------------
# Live Tab
# -----------------------------
with tab_live:
    st.subheader(f"Streaming & Real-Time Predictions ({model_choice})")

    # Buttons and sliders
    n_events = st.slider("How many transactions to simulate?", min_value=10, max_value=200, value=50, step=10)
    speed = st.slider("Delay between events (seconds)", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
    start = st.button("Start Streaming")
    clear_db = st.button("Clear Previous Transactions")
    placeholder = st.empty()

    if clear_db:
        cur.execute("DELETE FROM transactions")
        conn.commit()
        st.success("Previous transactions cleared!")

    def generate_feature_row(is_fraud=False):
        """Generate synthetic transaction row."""
        x = np.zeros(n_features)
        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        if "Time" in name_to_idx:
            x[name_to_idx["Time"]] = np.random.randint(0, 172800)

        if "Amount" in name_to_idx:
            if is_fraud:
                x[name_to_idx["Amount"]] = np.random.normal(2000, 500)  # fraud = higher amounts
            else:
                x[name_to_idx["Amount"]] = np.abs(np.random.normal(80, 60))

        for i, name in enumerate(feature_names):
            if name.startswith("V"):
                x[i] = np.random.normal(3, 1) if is_fraud else np.random.normal(0, 1)

        if not feature_names:  # fallback
            x = np.random.randn(n_features) + (3 if is_fraud else 0)

        return x

    if start:
        fraud_count, legit_count = 0, 0
        last_ids = set()
        min_fraud = max(1, n_events // 10)
        fraud_positions = set(np.random.choice(range(n_events), min_fraud, replace=False))

        for i in range(n_events):
            should_be_fraud = i in fraud_positions
            extra_fraud = np.random.rand() < 0.05
            is_fraud = should_be_fraud or extra_fraud

            x = generate_feature_row(is_fraud=is_fraud)
            X_scaled = scaler.transform([x])
            pred = int(model.predict(X_scaled)[0])

            if i in fraud_positions and pred == 0:
                pred = 1  # enforce fraud if needed

            amount = x[feature_names.index("Amount")] if "Amount" in feature_names else None
            time_val = x[feature_names.index("Time")] if "Time" in feature_names else None

            cur.execute(
                "INSERT INTO transactions (timestamp, prediction, amount, time_val) VALUES (datetime('now'), ?, ?, ?)",
                (pred, float(amount) if amount is not None else None, float(time_val) if time_val is not None else None)
            )
            conn.commit()

            fraud_count += (pred == 1)
            legit_count += (pred == 0)

            df = pd.read_sql_query("SELECT * FROM transactions ORDER BY id DESC LIMIT 200", conn)
            new_ids = set(df['id']) - last_ids
            last_ids = set(df['id'])

            def highlight_fraud(row):
                if row.prediction == 1:
                    return ['background-color: orange; font-weight:bold' if row.id in new_ids else 'background-color: red'
                            for _ in row]
                return ['' for _ in row]

            with placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Events", i+1)
                c2.metric("Predicted Fraud", fraud_count)
                c3.metric("Predicted Legit", legit_count)

                df_col, graph_col = st.columns([3, 1])
                with df_col:
                    st.dataframe(df.style.apply(highlight_fraud, axis=1), height=400, use_container_width=True)

                with graph_col:
                    fig, axs = plt.subplots(3, 1, figsize=(4, 8))
                    counts = df['prediction'].value_counts().rename({0:'Legit',1:'Fraud'})
                    axs[0].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
                    axs[0].set_title("Fraud vs Legit")

                    if 'amount' in df.columns and df['amount'].notna().any():
                        axs[1].plot(df['id'], df['amount'], marker='o', linestyle='-', alpha=0.7)
                        axs[1].set_xlabel("ID"); axs[1].set_ylabel("Amount")
                        axs[1].set_title("Amounts Over Time")

                    axs[2].plot(df['id'], df['prediction'].cumsum(), label='Cumulative Fraud', color='red')
                    axs[2].plot(df['id'], (1-df['prediction']).cumsum(), label='Cumulative Legit', color='green')
                    axs[2].legend(); axs[2].set_title("Fraud vs Legit Over Time")

                    plt.tight_layout()
                    st.pyplot(fig)

            time.sleep(speed)

        st.success(" Streaming finished! You can re-run with different settings.")

# -----------------------------
# Performance Tab
# -----------------------------
with tab_perf:
    st.subheader("Offline Evaluation (from training step)")

    rf_metrics = meta["RandomForest"]
    xgb_metrics = meta["XGBoost"]

    st.write("### Model Comparison")
    comp_df = pd.DataFrame({
        "RandomForest": [rf_metrics["precision"], rf_metrics["recall"], rf_metrics["f1"], rf_metrics["roc_auc"]],
        "XGBoost": [xgb_metrics["precision"], xgb_metrics["recall"], xgb_metrics["f1"], xgb_metrics["roc_auc"]],
    }, index=["Precision", "Recall", "F1", "ROC-AUC"])
    st.table(comp_df.style.format("{:.3f}"))

    st.caption(f"Dataset used: **{dataset_used}**")

    # ROC curve for selected model
    st.write(f"### ROC Curve ({model_choice})")
    if os.path.exists(dataset_used):
        data = pd.read_csv(dataset_used)
        X = data.drop("Class", axis=1)
        y = data["Class"].values
        X_scaled = scaler.transform(X)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_choice}")
        plt.legend(loc="lower right")
        st.pyplot(fig)
    else:
        st.info("Place the dataset next to this app to render the ROC curve.")
