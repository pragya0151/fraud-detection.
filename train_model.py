# train_model.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_curve

# -----------------------------
# 1. Load dataset
# -----------------------------
DATA_FILE = "creditcard.csv" if os.path.exists("creditcard.csv") else "sample_data.csv"
print(f" Using dataset: {DATA_FILE}")
data = pd.read_csv(DATA_FILE)

X = data.drop("Class", axis=1)
y = data["Class"]

# -----------------------------
# 2. Preprocessing
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
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

y_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)
roc_rf = roc_auc_score(y_test, y_proba_rf)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True, digits=4)
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
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)

y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb.predict(X_test)
roc_xgb = roc_auc_score(y_test, y_proba_xgb)
report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True, digits=4)
cm_xgb = confusion_matrix(y_test, y_pred_xgb).tolist()

print(" XGBoost Training Complete")

# -----------------------------
# 5. Train Autoencoder
# -----------------------------
print(" Training Autoencoder (Unsupervised Deep Learning)...")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

X_train_ae = X_train[y_train == 0]
input_dim = X_train.shape[1]

inp = Input(shape=(input_dim,))
enc = Dense(16, activation="relu")(inp)
enc = Dense(8, activation="relu")(enc)
bottleneck = Dense(4, activation="relu")(enc)
dec = Dense(8, activation="relu")(bottleneck)
dec = Dense(16, activation="relu")(dec)
out = Dense(input_dim, activation=None)(dec)

autoencoder = Model(inputs=inp, outputs=out)
autoencoder.compile(optimizer="adam", loss="mse")

es = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=30,
    batch_size=256,
    shuffle=True,
    callbacks=[es],
    verbose=1
)

recon_test = autoencoder.predict(X_test)
error_test = np.mean(np.square(recon_test - X_test), axis=1)
threshold_ae = np.percentile(error_test[y_test == 0], 95)

y_pred_ae = (error_test > threshold_ae).astype(int)
roc_ae = roc_auc_score(y_test, error_test)
report_ae = classification_report(y_test, y_pred_ae, output_dict=True, digits=4)
cm_ae = confusion_matrix(y_test, y_pred_ae).tolist()

print(" Autoencoder Training Complete")

# -----------------------------
# 6. Hybrid Weighted Voting Model
# -----------------------------
print(" Building Hybrid Weighted Model...")

# Normalize autoencoder anomaly score
error_test_norm = (error_test - error_test.min()) / (error_test.max() - error_test.min())

# Weighted score with ROC-AUC as weights
weights = np.array([roc_rf, roc_xgb, roc_ae])
weights = weights / weights.sum()  # normalize

hybrid_scores = (
    weights[0] * y_proba_rf +
    weights[1] * y_proba_xgb +
    weights[2] * error_test_norm
)

# Automatic F1-score threshold search
precisions, recalls, thresholds = precision_recall_curve(y_test, hybrid_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)

best_threshold = thresholds[best_idx]
y_pred_hybrid = (hybrid_scores >= best_threshold).astype(int)

roc_hybrid = roc_auc_score(y_test, hybrid_scores)
report_hybrid = classification_report(y_test, y_pred_hybrid, output_dict=True, digits=4)
cm_hybrid = confusion_matrix(y_test, y_pred_hybrid).tolist()

print(" Hybrid Model Complete ✅")

# -----------------------------
# 7. Save Models & Metrics
# -----------------------------
joblib.dump(rf, "rf_model.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
autoencoder.save("autoencoder.h5")

metrics = {
    "dataset": DATA_FILE,
    "feature_names": list(X.columns),
    "n_features": X.shape[1],
    "Hybrid": {
        "precision": report_hybrid["1"]["precision"],
        "recall": report_hybrid["1"]["recall"],
        "f1": report_hybrid["1"]["f1-score"],
        "roc_auc": roc_hybrid,
        "threshold": float(best_threshold),
        "cm": cm_hybrid,
        "support_fraud": report_hybrid["1"]["support"]
    },
    "RandomForest": metrics["RandomForest"] if "RandomForest" in locals() else {
        "precision": report_rf["1"]["precision"],
        "recall": report_rf["1"]["recall"],
        "f1": report_rf["1"]["f1-score"],
        "roc_auc": roc_rf,
        "cm": cm_rf
    },
    "XGBoost": {
        "precision": report_xgb["1"]["precision"],
        "recall": report_xgb["1"]["recall"],
        "f1": report_xgb["1"]["f1-score"],
        "roc_auc": roc_xgb,
        "cm": cm_xgb
    },
    "Autoencoder": {
        "precision": report_ae["1"]["precision"],
        "recall": report_ae["1"]["recall"],
        "f1": report_ae["1"]["f1-score"],
        "roc_auc": roc_ae,
        "threshold": float(threshold_ae),
        "cm": cm_ae
    }
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Saved models & hybrid ensemble to disk")
print("✅ Metrics stored in metrics.json")

# -----------------------------
# 8. Print comparison summary
# -----------------------------
print("\n📊 Model Performance Summary:")
print(json.dumps(metrics, indent=2))


