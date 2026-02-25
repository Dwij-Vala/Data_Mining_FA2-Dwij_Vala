# ============================================================
# ATM Intelligence Demand Forecasting - FA-2
# Complete Free-Flow Analytical Application (Styled Version)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(page_title="ATM Intelligence Demand Forecasting",
                   layout="wide")

# ------------------------------------------------------------
# Custom Gradient Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 40%,
            #0f172a 100%
        );
    }

    h1, h2, h3 {
        color: #e2e8f0;
    }

    p, div {
        color: #cbd5e1;
    }

    section[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 12px;
    }

    hr {
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_atm_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("ATM Intelligence Demand Forecasting – FA-2")
st.write(
    "This application transforms preprocessed ATM transaction data into actionable insights "
    "using Exploratory Data Analysis (EDA), Clustering, and Anomaly Detection."
)

st.markdown("---")

# ------------------------------------------------------------
# 1. Dataset Overview
# ------------------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(df))
col2.metric("Number of ATMs", df["ATM_ID"].nunique())
col3.metric("Date Range (Days)", (df["Date"].max() - df["Date"].min()).days)
col4.metric("Average Withdrawals", round(df["Total_Withdrawals"].mean(), 2))

st.write("Sample Data Preview:")
st.dataframe(df.head())

st.markdown("---")

# ------------------------------------------------------------
# 2. Exploratory Data Analysis
# ------------------------------------------------------------

st.header("2. Exploratory Data Analysis")

# Distribution
st.subheader("Distribution of Withdrawals and Deposits")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(df["Total_Withdrawals"], bins=30, kde=True, ax=ax)
    ax.set_title("Total Withdrawals Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(df["Total_Deposits"], bins=30, kde=True, ax=ax)
    ax.set_title("Total Deposits Distribution")
    st.pyplot(fig)

st.write(
    "Observation: Withdrawals show natural variability with occasional high-value spikes "
    "indicating increased demand during special days."
)

# Time Trend
st.subheader("Withdrawal Trends Over Time")

daily_avg = df.groupby("Date")["Total_Withdrawals"].mean()

fig, ax = plt.subplots(figsize=(10,4))
daily_avg.plot(ax=ax)
ax.set_title("Average Withdrawals Over Time")
st.pyplot(fig)

st.write(
    "Observation: Periodic spikes suggest weekly demand cycles and potential holiday impacts."
)

# Holiday Impact
st.subheader("Impact of Holidays and Special Events")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.barplot(x="Holiday_Flag", y="Total_Withdrawals", data=df, ax=ax)
    ax.set_title("Holiday vs Non-Holiday Withdrawals")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.barplot(x="Special_Event_Flag", y="Total_Withdrawals", data=df, ax=ax)
    ax.set_title("Event vs Normal Day Withdrawals")
    st.pyplot(fig)

st.markdown("---")

# ------------------------------------------------------------
# 3. Clustering Analysis
# ------------------------------------------------------------

st.header("3. Clustering ATMs Based on Demand Behavior")

features = df[[
    "Total_Withdrawals",
    "Total_Deposits",
    "Location_Type",
    "Nearby_Competitor_ATMs"
]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow Method
st.subheader("Elbow Method")

inertia = []
k_range = range(1, 7)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_features)
    inertia.append(model.inertia_)

fig, ax = plt.subplots()
ax.plot(k_range, inertia, marker="o")
ax.set_xlabel("Clusters")
ax.set_ylabel("Inertia")
st.pyplot(fig)

# Silhouette
st.subheader("Silhouette Score")

sil_scores = []
for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(scaled_features)
    sil_scores.append(silhouette_score(scaled_features, labels))

fig, ax = plt.subplots()
ax.plot(range(2, 7), sil_scores, marker="o")
ax.set_xlabel("Clusters")
ax.set_ylabel("Silhouette Score")
st.pyplot(fig)

# Apply KMeans
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_features)

# PCA Visualization
st.subheader("Cluster Visualization (PCA Reduced)")

pca = PCA(n_components=2)
components = pca.fit_transform(scaled_features)

fig, ax = plt.subplots()
scatter = ax.scatter(components[:, 0], components[:, 1], c=df["Cluster"])
ax.set_title("Cluster Separation")
st.pyplot(fig)

st.write(
    "Cluster Interpretation:\n"
    "- Cluster 0: High Demand Urban ATMs\n"
    "- Cluster 1: Stable Business Zone ATMs\n"
    "- Cluster 2: Low Demand Rural ATMs"
)

st.markdown("---")

# ------------------------------------------------------------
# 4. Anomaly Detection
# ------------------------------------------------------------

st.header("4. Anomaly Detection Using Isolation Forest")

iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(df[["Total_Withdrawals"]])

fig, ax = plt.subplots(figsize=(10,4))
colors = df["Anomaly"].map({1: "skyblue", -1: "red"})
ax.scatter(df["Date"], df["Total_Withdrawals"], c=colors)
ax.set_title("Withdrawal Anomalies (Red = Anomaly)")
st.pyplot(fig)

st.write(
    "Red points represent unusual withdrawal spikes that may require operational attention."
)

st.markdown("---")

# ------------------------------------------------------------
# 5. Interactive Demand Planner
# ------------------------------------------------------------

st.header("5. Interactive Demand Planner")

location_choice = st.selectbox("Select Location Type", df["Location_Type"].unique())
day_choice = st.selectbox("Select Day of Week", df["Day_of_Week"].unique())
holiday_choice = st.selectbox("Holiday?", df["Holiday_Flag"].unique())

filtered = df[
    (df["Location_Type"] == location_choice) &
    (df["Day_of_Week"] == day_choice) &
    (df["Holiday_Flag"] == holiday_choice)
]

st.write("Filtered Data Preview")
st.dataframe(filtered.head())

if len(filtered) > 0:
    avg_withdrawal = filtered["Total_Withdrawals"].mean()
    st.metric("Estimated Average Withdrawal", round(avg_withdrawal, 2))

    if avg_withdrawal > df["Total_Withdrawals"].mean():
        st.success("Demand Level: HIGH – Increase cash allocation.")
    else:
        st.info("Demand Level: NORMAL – Standard allocation sufficient.")
else:
    st.warning("No matching data found.")

st.markdown("---")

st.write("FA-2 Analytical Workflow Completed Successfully.")