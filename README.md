# Real-Time Fraud Detection

## This project demonstrates how real-time fraud detection works in fintech companies:

- Transactions continuously arrive  
- Machine learning and deep learning models classify suspicious transactions  
- A dashboard updates in real time for analysts to monitor fraud  
- All predictions are stored in a database for further investigation  

It is designed for students, researchers, and professionals interested in streaming machine learning systems used in payment fraud detection.

---

## FAQ

<details>
<summary>Why multiple models are used?</summary>

- RandomForest is easy to train and provides stable precision  
- XGBoost improves recall and catches more fraud patterns  
- Autoencoder detects unusual behavior not seen during training  
- Hybrid Ensemble combines all three for the best overall results  

Model performance in this project:

RandomForest → Precision: 0.961, Recall: 0.755, ROC-AUC: 0.957  
XGBoost → Precision: 0.891, Recall: 0.837, ROC-AUC: 0.975  
Autoencoder → Precision: 0.028, Recall: 0.847, ROC-AUC: 0.935  
Hybrid → Precision: 0.942, Recall: 0.827, ROC-AUC: 0.967  

Conclusion: The Hybrid model is the recommended final model because it balances fraud detection ability and false alarms better than individual models.
</details>

---

<details>
<summary>Do I need the Kaggle dataset to run this?</summary>

Not required. The app generates synthetic transactions for real-time testing.  
However, providing the dataset allows better model evaluation and retraining.
</details>

---

<details>
<summary>Why SQLite for storage?</summary>

SQLite is lightweight, file-based, and ideal for demo projects.  
In production, high-volume streaming systems use PostgreSQL, Cassandra, Kafka, etc.
</details>

---

<details>
<summary>Can this scale to real banks?</summary>

This is a prototype implementation.  
In real deployments, additional elements are used such as  
deep learning at scale, graph-based fraud detection, and big-data streaming pipelines.
</details>

---

<details>
<summary>Why Autoencoder has low precision?</summary>

Autoencoder learns normal transaction patterns only.  
Any unusual deviation is flagged as fraud, including some genuine cases.  
This causes lower precision but improves unseen fraud detection.  
It is valuable when new fraud patterns appear that were not part of training data.
</details>

---

<details>
<summary>What is the difference between precision and recall in fraud detection?</summary>

Precision: Out of all flagged frauds, how many are truly fraud  
Recall: Out of all actual frauds, how many did the model detect  

In fraud detection, high recall is more important because missing fraud causes financial loss.
</details>

---

## Key Features

- Real-time transaction streaming with adjustable speed and volume  
- Machine learning + deep anomaly detection for fraud detection  
  - RandomForest  
  - XGBoost  
  - Autoencoder  
  - Hybrid Ensemble Model  
- Live dashboard visualizations  
  - Fraud vs Legit distribution  
  - Amount trend  
  - Cumulative fraud monitoring  
- Offline performance comparison including ROC-AUC, Precision, Recall, F1  
- SQLite database storage for transaction history  
- Synthetic data generator included  

---

## Tech Stack

- Python 3.8+  
- Streamlit  
- Scikit-learn  
- XGBoost  
- TensorFlow / Keras Autoencoder  
- SQLite3  
- Pandas, NumPy, Matplotlib  

---

## App Features

1. Live Transaction Monitoring  
   Fraud transactions highlighted visually  
   Real-time visualizations update continuously  
  <img width="1694" height="867" alt="image" src="https://github.com/user-attachments/assets/319101b9-eb39-4af2-aab1-7239a64678fe" />
  <img width="1288" height="758" alt="image" src="https://github.com/user-attachments/assets/c7265f09-56af-4635-8e85-56173b29b233" />



2. Model Performance Comparison  
    table includes 4 models with improvements explained  
   ROC-AUC and Precision-Recall evaluation  
   <img width="1315" height="701" alt="image" src="https://github.com/user-attachments/assets/b9bc5020-ab86-4980-8b23-84178d225862" />
   <img width="1299" height="793" alt="image" src="https://github.com/user-attachments/assets/600dde14-234b-49b1-a9a6-9d5cb3cd2f8e" />
   <img width="1266" height="366" alt="image" src="https://github.com/user-attachments/assets/b19a7a04-4faf-4d31-8b2a-c99cdecdba27" />
   <img width="834" height="638" alt="image" src="https://github.com/user-attachments/assets/03e2cbb9-2558-4725-b9f8-5a00a68ee600" />





3. Clear Database Reset Function  
   One-click action to clear all transaction logs
   <img width="1287" height="243" alt="image" src="https://github.com/user-attachments/assets/9c3aebed-a77c-46a1-bb6d-5674e2702d43" />


5. Streaming Configuration Controls  
   Adjustable interval and number of simulated transactions  

6. Model Selection for Live Predictions  
   Choose RandomForest, XGBoost, Autoencoder, or Hybrid Ensemble  
   and compare behaviors instantly
   <img width="408" height="368" alt="image" src="https://github.com/user-attachments/assets/016cd914-52b4-444e-988c-76da1bac65b4" />


8. Sidebar FAQ and System Description  
   Clear explanations for performance differences  
   and model behavior reasoning
   <img width="442" height="529" alt="image" src="https://github.com/user-attachments/assets/c7da53cc-b3c9-4930-84de-40ea5cd3ab41" />
   <img width="873" height="590" alt="image" src="https://github.com/user-attachments/assets/96b5528a-c475-4a62-8eec-83117c3d6871" />



---

## Dataset Information

The original credit card fraud dataset is not included due to size limitations.  
Download here:

https://www.kaggle.com/mlg-ulb/creditcardfraud

Place the file in the root folder and run `train_model.py` to retrain models.  
If not available, the app still runs using synthetic data.

---

---

##  Workflow

<img width="634" height="846" alt="image" src="https://github.com/user-attachments/assets/e5a43700-877e-4982-8d6f-08387d772c7d" />

---

##  Project Structure
<img width="367" height="435" alt="image" src="https://github.com/user-attachments/assets/9b89c6b0-3f6e-4086-ab86-aab7826df67e" />







