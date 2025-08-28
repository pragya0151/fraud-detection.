# Real-Time Fraud Detection  

## This project demonstrates **how real-time fraud detection works** in fintech companies:  

- Transactions continuously arrive.  
- ML models (RandomForest & XGBoost) flag suspicious ones.  
- A dashboard updates in **real time** for analysts to monitor fraud.  
- Stored in a **database** for further investigation.  

It is designed for **students, researchers, and professionals** who want to explore **streaming ML + fintech fraud systems** in practice.    

---
##  FAQ  

<details>
<summary> Why use both RandomForest and XGBoost?</summary>

- **RandomForest** is easy to train, interpretable, and fast.  
- **XGBoost** is more powerful, handles imbalanced data better, and often achieves higher ROC-AUC.  

 Example (from this project):  
- RandomForest → Precision: 0.96, Recall: 0.75, ROC-AUC: 0.95  
- XGBoost → Precision: 0.89, Recall: 0.83, ROC-AUC: 0.97  

 Conclusion: **XGBoost is better overall** but RandomForest is simpler & faster. Both are included for comparison.  
</details>

---

<details>
<summary> Do I need the Kaggle dataset to run this?</summary>
No. The app includes a **synthetic transaction generator**, so you can run it without the dataset.  
However, for best results, download the dataset and train the models.  
</details>

---

<details>
<summary> Why SQLite for storage?</summary>
SQLite is lightweight, file-based, and perfect for demos.  
In production, banks use **PostgreSQL, Cassandra, or Kafka** for large-scale fraud detection.  
</details>

---

<details>
<summary> Can this scale to real banks?</summary>
This project is a **prototype**.  
In production, additional features like **deep learning, graph-based fraud detection, and big data streaming (Kafka, Spark)** are used.  
</details>

---

<details>
<summary> What is the difference between precision and recall in fraud detection?</summary>

- **Precision** → Out of all flagged frauds, how many are truly fraud?  
- **Recall** → Out of all actual frauds, how many did the system catch?  

 In fraud detection, **high recall is crucial** (better to flag more frauds, even with some false alarms).  
</details>

##  Key Features

- **Real-time Transaction Simulation:** Adjustable batch size and streaming speed.  
- **Fraud Detection Models:**  
  - **RandomForest** – fast, interpretable, and easy to train.  
  - **XGBoost** – more accurate and optimized for imbalanced data.  
- **Interactive Dashboard:**  
  - Fraudulent transactions highlighted in live data table.  
  - Visualizations: Pie chart, line charts, cumulative fraud trends.  
- **Offline Model Comparison:** Precision, Recall, F1-score, ROC-AUC, and ROC curve for both models.  
- **Database Management:** Stores all transactions in **SQLite** with a one-click option to clear history.  
- **Synthetic Data Generation:** Works even without downloading the dataset.  

---

##  Tech Stack

- **Python 3.8+**  
- **Streamlit** → Interactive UI & real-time dashboard  
- **Scikit-learn** → RandomForest classifier  
- **XGBoost** → Gradient-boosted decision trees for high performance  
- **SQLite3** → Transaction storage  
- **Pandas & Matplotlib** → Data processing & visualization  

---

##  App Features

1. **Live Transaction Dashboard**  
   - Fraud transactions are highlighted in the table.  
   - Mini-graphs update in real time.
     <img width="1900" height="928" alt="image" src="https://github.com/user-attachments/assets/b95a641c-f874-461e-a820-510ad229e4f8" />
     <img width="1852" height="866" alt="image" src="https://github.com/user-attachments/assets/4037bf1e-81cf-4cbd-b048-06b46a4dda61" />
     <img width="630" height="857" alt="image" src="https://github.com/user-attachments/assets/aa8a4ca9-11d6-4bc3-89d8-4a1b4fb4620a" />




2. **Model Performance Comparison**  
   - Metrics (Precision, Recall, F1, ROC-AUC) for RandomForest & XGBoost.  
   - ROC curve visualization.
   - <img width="1853" height="747" alt="image" src="https://github.com/user-attachments/assets/8a534ae5-7cb6-4cd8-8bd0-da0c6610f34a" />
   <img width="1330" height="876" alt="image" src="https://github.com/user-attachments/assets/b5c1f816-d4d3-4d21-bf26-d7a3be3a8b68" />



3. **Clear Transactions Button**  
   - Resets the database for new runs.
   - <img width="1749" height="736" alt="image" src="https://github.com/user-attachments/assets/144fff7a-baa4-48b5-851f-958e4ffaedf8" />


4. **Streaming Simulation Controls**  
   - Adjust speed & batch size to mimic real payment systems.


     
5. **choose between both model to see for yourself**
   -<img width="509" height="252" alt="image" src="https://github.com/user-attachments/assets/ebaf6a12-4dff-4da7-a94b-22eb89c16b3e" />
   
   
6.**about website and faq section**
    - <img width="429" height="924" alt="image" src="https://github.com/user-attachments/assets/b592b9d1-5f11-44a6-ab83-c115adc4c57b" />

    
    -<img width="368" height="846" alt="image" src="https://github.com/user-attachments/assets/7547d8f3-c26d-472b-8ae1-65acec696fdd" />



---

##  Dataset Information
The original **credit card dataset** is **not included** due to GitHub size restrictions (~144 MB).  

 Download here: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
Place it in the project folder to retrain models.  
If not available, the app **auto-generates synthetic transactions** so you can still use it.  

---

##  Workflow

<img width="634" height="846" alt="image" src="https://github.com/user-attachments/assets/e5a43700-877e-4982-8d6f-08387d772c7d" />

---

##  Project Structure
<img width="602" height="592" alt="image" src="https://github.com/user-attachments/assets/8aa9f0e7-cdf8-402e-b1a9-32ab4bd2c269" />






