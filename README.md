# 🏦 ATM Intelligence Demand Forecasting – FA-2

## 📌 Student Details

**Name:** Dwij Vala  
**Programme:** IBCP – Year 1  
**Course:** CRS – Artificial Intelligence  
**Module:** Data Mining  
**Student ID:** 2505369  

---

## 📌 Project Overview

This project was developed as part of the CRS – Artificial Intelligence (Data Mining) curriculum.

The objective of this application is to transform preprocessed ATM transaction data into actionable insights using:

- Exploratory Data Analysis (EDA)
- Clustering Techniques (K-Means)
- Anomaly Detection (Isolation Forest)
- Interactive Demand Planning

The application is built using **Python and Streamlit** and follows a structured, free-flow analytical workflow aligned with FA-2 assessment requirements.

---

## 🎯 Problem Statement

FinTrust Bank manages a nationwide ATM network and faces operational challenges:

- Some ATMs run out of cash during peak demand periods (festivals, salary days).
- Some ATMs remain overstocked, increasing idle cash costs.

This project applies data mining techniques to uncover demand patterns, group ATMs based on behavior, detect unusual spikes, and support informed cash allocation decisions.

---

## 🚀 Live Application

🔗 **Streamlit App Link:**  
https://atm-app-dwij.streamlit.app/

---

## 📊 Dataset Description

The dataset used in this project is synthetically generated for analytical simulation and includes:

| Column Name | Description |
|------------|-------------|
| ATM_ID | Unique ATM identifier |
| Date | Transaction date |
| Day_of_Week | Day index (0–6) |
| Time_of_Day | Time category (1–4) |
| Total_Withdrawals | Total withdrawals per ATM per day |
| Total_Deposits | Total deposits per ATM per day |
| Previous_Day_Cash_Level | Available cash from previous day |
| Location_Type | 1=Urban, 2=Semi-Urban, 3=Rural |
| Holiday_Flag | 1 if holiday |
| Special_Event_Flag | 1 if event day |
| Weather_Condition | Weather category |
| Nearby_Competitor_ATMs | Competitor presence (0/1) |
| Cash_Demand_Next_Day | Target variable |

Dataset Summary:
- 60 ATMs  
- 180 Days  
- 10,800 Records  

---

## 🔎 Exploratory Data Analysis (EDA)

The EDA section includes:

- Distribution analysis (histograms and boxplots)
- Time-series withdrawal trends
- Holiday and special event impact comparison
- Correlation heatmap
- Pattern identification and spike detection

### Key Observations

- Withdrawal spikes occur on weekends and holidays.
- Urban ATMs show higher volatility.
- Special events significantly increase withdrawal activity.
- Demand patterns exhibit periodic weekly trends.

---

## 🧠 Clustering Analysis

Clustering was performed using **K-Means** after feature standardization.

### Process

1. Feature selection  
2. StandardScaler normalization  
3. Elbow Method evaluation  
4. Silhouette score validation  
5. PCA-based cluster visualization  

### Identified ATM Segments

- **Cluster 0:** High Demand Urban ATMs  
- **Cluster 1:** Stable Business Zone ATMs  
- **Cluster 2:** Low Demand Rural ATMs  

This grouping enables strategic cash allocation planning and operational optimization.

---

## 🚨 Anomaly Detection

Anomalies were detected using **Isolation Forest**.

- Identifies unusual withdrawal spikes  
- Highlights abnormal demand behavior  
- Supports proactive replenishment planning  

Red-highlighted points in visualizations represent detected anomalies.

---

## 🧮 Interactive Demand Planner

The application includes an interactive decision-support section where users can:

- Filter by Location Type  
- Select Day of Week  
- Choose Holiday status  

The planner calculates expected average withdrawal levels and suggests whether increased cash allocation is required.

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📁 Project Structure

ATM_FA2_Project/
│
├── app.py
├── cleaned_atm_data.csv
├── generate_data.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies

pip install -r requirements.txt

### 2️⃣ Run the Application

python -m streamlit run app.py

The application will open automatically in your browser.

---

## 📌 Academic Alignment

This project fulfills all FA-2 rubric requirements:

✔ Comprehensive EDA with interpretation  
✔ Proper clustering with validation techniques  
✔ Accurate anomaly detection  
✔ End-to-end reproducible Python script  
✔ Interactive analytical workflow  
✔ Clear documentation and structured implementation  

---

## 📘 Academic Submission Note

This project demonstrates the transformation of prepared data (FA-1) into actionable analytical insights through applied data mining techniques in FA-2.
