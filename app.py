
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# -------- PAGE CONFIGURATION --------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------- LOAD DATASET --------


@st.cache_data
def load_data():
    return pd.read_csv("data/Mall_Customers.csv")


df = load_data()

# -------- TRAIN K-MEANS MODEL --------

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

# -------- NAME CUSTOMER SEGMENTS --------

segment_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "High Spending Low Income",
    3: "Careful High Income",
    4: "Budget Conscious Customers"
}

df["Segment"] = df["Cluster"].map(segment_names)

# -------- DASHBOARD TITLE --------

st.title("📊 Customer Segmentation Dashboard")

st.write(
    "An interactive dashboard that groups customers "
    "based on their annual income and spending behaviour."
)

st.divider()

# -------- KEY METRICS --------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    st.metric("Customer Segments", df["Cluster"].nunique())

with col3:
    st.metric(
        "Average Income",
        f"${df['Annual Income (k$)'].mean():.2f}k"
    )

with col4:
    st.metric(
        "Average Spending",
        f"{df['Spending Score (1-100)'].mean():.2f}"
    )

st.divider()

# -------- CUSTOMER SEGMENT GRAPH --------

st.subheader("Customer Segments")

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Segment",
    palette="Set2",
    s=100,
    ax=ax
)

ax.set_title("Customer Segmentation using K-Means")
ax.set_xlabel("Annual Income (in thousands)")
ax.set_ylabel("Spending Score")

st.pyplot(fig)

# -------- SEGMENT FILTER --------

st.subheader("Explore Customer Segments")

selected_segment = st.selectbox(
    "Choose a customer segment",
    ["All Segments"] + list(segment_names.values())
)

if selected_segment == "All Segments":
    filtered_df = df
else:
    filtered_df = df[df["Segment"] == selected_segment]

st.write(f"Customers in selection: {len(filtered_df)}")

st.dataframe(
    filtered_df[
        [
            "CustomerID",
            "Genre",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Segment"
        ]
    ],
    use_container_width=True
)

# -------- SEGMENT SUMMARY --------

st.subheader("Customer Segment Summary")

summary = df.groupby("Segment").agg(
    Number_of_Customers=("CustomerID", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

st.dataframe(summary, use_container_width=True)
