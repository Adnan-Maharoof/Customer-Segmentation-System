
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# -------- LOAD DATASET --------

df = pd.read_csv("data/Mall_Customers.csv")

print(df.head())
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())


# -------- EDA (Exploratory Data Analysis) --------

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())


# -------- CUSTOMER AGE DISTRIBUTION --------

plt.figure(figsize=(8, 5))

sns.histplot(data=df, x="Age", bins=15, kde=True)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number Of Customers")

plt.tight_layout()
plt.show()


# -------- ANNUAL INCOME VS SPENDING SCORE --------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    s=80
)

plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()


# -------- SELECT FEATURES FOR K-MEANS --------

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


# -------- ELBOW METHOD --------

wcss = []

for i in range(1, 11):

    elbow_model = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    elbow_model.fit(X)

    wcss.append(elbow_model.inertia_)


# -------- PLOT ELBOW GRAPH --------

plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.tight_layout()
plt.show()


# -------- FINAL K-MEANS MODEL --------

optimal_clusters = 5

final_kmeans = KMeans(
    n_clusters=optimal_clusters,
    init="k-means++",
    random_state=42,
    n_init=10
)

# Train the final model
final_kmeans.fit(X)

# Assign cluster labels to customers
df["Cluster"] = final_kmeans.labels_

print("\n--- Customer Segments ---")
print(df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())


# -------- VISUALIZE CUSTOMER SEGMENTS --------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set2",
    s=100
)

plt.title("Customer Segments using K-Means")
plt.xlabel("Annual Income (in thousands)")
plt.ylabel("Spending Score")

plt.legend(title="Customer Cluster")
plt.tight_layout()
plt.show()

# -------- ANALYSE CUSTOMER SEGMENTS --------

segment_summary = df.groupby("Cluster").agg(
    Number_of_Customers=("Cluster", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

print("\n--- Customer Segment Analysis ---")
print(segment_summary)


# -------- NAME CUSTOMER SEGMENTS --------

segment_names = {
    0: "Average Customers",
    1: "Premium Customers",
    2: "High Spending Low Income",
    3: "Careful High Income",
    4: "Budget Conscious Customers"
}

df["Segment"] = df["Cluster"].map(segment_names)

print("\n--- Final Customer Segments ---")

print(df[[
    "CustomerID",
    "Annual Income (k$)",
    "Spending Score (1-100)",
    "Cluster",
    "Segment"
]].head(10))
