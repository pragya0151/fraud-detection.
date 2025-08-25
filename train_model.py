import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

DATA_FILE = "creditcard.csv" if os.path.exists("creditcard.csv") else "sample_data.csv"

print(f"📦 Using dataset: {DATA_FILE}")
data = pd.read_csv(DATA_FILE)

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
report = classification_report(y_test, y_pred, output_dict=True, digits=4)
roc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred).tolist()

print("✅ Training complete")
print(json.dumps({
    "precision": report["1"]["precision"],
    "recall": report["1"]["recall"],
    "f1": report["1"]["f1-score"],
    "roc_auc": roc,
    "support_fraud": report["1"]["support"]
}, indent=2))

joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("metrics.json", "w") as f:
    json.dump({
        "feature_names": list(X.columns),
        "n_features": X.shape[1],
        "metrics": {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "roc_auc": roc,
            "cm": cm
        },
        "dataset": DATA_FILE
    }, f, indent=2)

print("💾 Saved rf_model.pkl, scaler.pkl, metrics.json")
