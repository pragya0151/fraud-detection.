# Real-Time Fraud Detection 

##  Project Overview
This project is a **real-time fraud detection system** built with **Python, Streamlit, and RandomForest**. It simulates streaming financial transactions, detects fraudulent activities on the fly, and provides **interactive dashboards** to monitor trends in real-time.  
The system maintains transaction history in a **SQLite database**, highlights fraudulent transactions, and displays **offline evaluation metrics** using a pre-trained model.  

This project is designed to **mimic real-world financial fraud detection scenarios**, where transactions arrive continuously, and fraud must be identified quickly and efficiently.

---

## Key Features

- **Real-time Transaction Simulation:** Generate synthetic transactions with adjustable batch size and streaming speed.  
- **Fraud Detection:** Powered by a **RandomForest classifier** trained on credit card data.  
- **Interactive Highlighting:** Fraudulent transactions are highlighted in the data table for easy identification.  
- **Mini-Dashboard Visualizations:**  
  - **Pie chart:** Fraud vs Legit transactions  
  - **Line chart:** Transaction amounts over time  
  - **Line chart:** Cumulative fraud vs legit counts  
- **Offline Evaluation Metrics:** Precision, Recall, F1-score, ROC-AUC, and ROC curve.  
- **Database Management:** Stores all transactions in SQLite; includes a button to **clear previous transactions**.  
- **Synthetic Data Generation:** Ensures the app works even without the original dataset.  

---

## Tech Stack

- **Python 3.8+**  
- **Streamlit:** Interactive UI and dashboard  
- **Scikit-learn:** RandomForest model  
- **SQLite3:** Transaction storage  
- **Pandas & Matplotlib:** Data processing and visualization  

---

## App features



1. **Live Transaction Dashboard:**  
   - Table with highlighted frauds and mini-graphs.
   - <img width="1803" height="777" alt="Screenshot 2025-08-25 144649" src="https://github.com/user-attachments/assets/21aca6f8-42c2-4ca1-bf86-951ab833ece4" />

   - <img width="1193" height="768" alt="Screenshot 2025-08-25 144713" src="https://github.com/user-attachments/assets/a6b92fef-da47-41bd-ae43-3404367e7d2d" />
  
2. **Performance Tab / ROC Curve:**  
   - Offline metrics and ROC curve.
   - <img width="1318" height="897" alt="Screenshot 2025-08-25 145033" src="https://github.com/user-attachments/assets/64bafd67-9add-4d94-a8c6-b73e6d222b87" />
   <img width="1137" height="506" alt="Screenshot 2025-08-25 145022" src="https://github.com/user-attachments/assets/3666584a-71fb-49dc-9332-fbd3368e109c" />


3. **Clearing Transactions Feature:**  
   - Before and after clearing transactions.
   - <img width="1191" height="742" alt="Screenshot 2025-08-25 144757" src="https://github.com/user-attachments/assets/ad353e5a-1bad-44a6-bd17-209f3e8fd002" />
   <img width="1160" height="704" alt="Screenshot 2025-08-25 144828" src="https://github.com/user-attachments/assets/770799cc-68ba-4de4-b17a-d0077ad6590a" />

  
4. **Mini-Graphs Close-up (Optional):**  
   - Show readability and clarity of metrics.
   - <img width="588" height="879" alt="Screenshot 2025-08-25 144734" src="https://github.com/user-attachments/assets/d61dc76f-a686-44e3-94fe-f60d3dac3912" />

5. **Streaming Transactions Simulation:**  
   - Different batch size / speed to show real-time updates.
   - <img width="1191" height="742" alt="Screenshot 2025-08-25 144757" src="https://github.com/user-attachments/assets/a0805df7-9a84-4abf-934b-f30e9ad368a2" />




---

## Dataset Information

The original **credit card dataset** is **not included** due to GitHub file size limits (~144 MB).  

You can download it from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project folder to train the model.  
The app can **generate synthetic transaction data** if the dataset is not present, allowing you to run it without downloading the dataset.  

---
## workflow
<img width="1219" height="260" alt="image" src="https://github.com/user-attachments/assets/8ca622b5-d96f-4d60-86eb-550aad99f998" />

## project structure
realtime-fraud-rf/
│
├── app.py # Main Streamlit application
├── train_model.py # Model training script
├── rf_model.pkl # Trained RandomForest model
├── scaler.pkl # Feature scaler
├── metrics.json # Model evaluation metrics
├── realtime_fraud.db # SQLite database storing transactions
├── requirements.txt # Required Python packages
├── run_windows.bat # Batch script to run app on Windows
├── sample_data.csv # Optional sample data for testing
└── .gitignore # Git ignore file


