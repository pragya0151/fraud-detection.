import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_FILE = "creditcard.csv" if os.path.exists("creditcard.csv") else "sample_data.csv"
print(f" Using dataset: {DATA_FILE}")
data = pd.read_csv(DATA_FILE)

X = data.drop("Class", axis=1)
y = data["Class"]

# -----------------------------
# 2. Preprocessing (Scaling)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Train RandomForest
# -----------------------------
print(" Training RandomForest...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

report_rf = classification_report(y_test, y_pred_rf, output_dict=True, digits=4)
roc_rf = roc_auc_score(y_test, y_proba_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf).tolist()

print(" RandomForest Training Complete")

# -----------------------------
# 4. Train XGBoost
# -----------------------------
print(" Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # imbalance handling
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True, digits=4)
roc_xgb = roc_auc_score(y_test, y_proba_xgb)
cm_xgb = confusion_matrix(y_test, y_pred_xgb).tolist()

print(" XGBoost Training Complete")

# -----------------------------
# 5. Save Models & Metrics
# -----------------------------
joblib.dump(rf, "rf_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")

metrics = {
    "dataset": DATA_FILE,
    "feature_names": list(X.columns),
    "n_features": X.shape[1],
    "RandomForest": {
        "precision": report_rf["1"]["precision"],
        "recall": report_rf["1"]["recall"],
        "f1": report_rf["1"]["f1-score"],
        "roc_auc": roc_rf,
        "cm": cm_rf,
        "support_fraud": report_rf["1"]["support"]
    },
    "XGBoost": {
        "precision": report_xgb["1"]["precision"],
        "recall": report_xgb["1"]["recall"],
        "f1": report_xgb["1"]["f1-score"],
        "roc_auc": roc_xgb,
        "cm": cm_xgb,
        "support_fraud": report_xgb["1"]["support"]
    }
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved rf_model.pkl, xgb_model.pkl, scaler.pkl, and metrics.json")

# -----------------------------
# 6. Print comparison summary
# -----------------------------
print("\n Model Performance Summary:")
print(json.dumps({
    "RandomForest": {
        "Precision": metrics["RandomForest"]["precision"],
        "Recall": metrics["RandomForest"]["recall"],
        "F1": metrics["RandomForest"]["f1"],
        "ROC-AUC": metrics["RandomForest"]["roc_auc"]
    },
    "XGBoost": {
        "Precision": metrics["XGBoost"]["precision"],
        "Recall": metrics["XGBoost"]["recall"],
        "F1": metrics["XGBoost"]["f1"],
        "ROC-AUC": metrics["XGBoost"]["roc_auc"]
    }
}, indent=2))


