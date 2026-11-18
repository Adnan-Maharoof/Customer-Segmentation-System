# Customer Segmentation System

A machine learning project that uses **K-Means Clustering** to segment mall customers based on their annual income and spending score.

## Project Overview

The Customer Segmentation System groups customers into different segments based on their purchasing behaviour. This helps businesses understand their customers and make better marketing decisions.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Machine Learning Algorithm

**K-Means Clustering**

The model uses two features:
- Annual Income
- Spending Score

The Elbow Method was used to help determine the number of clusters. The final model uses 5 customer segments.

## Customer Segments

1. Average Customers
2. Premium Customers
3. High Spending Low Income
4. Careful High Income
5. Budget Conscious Customers

## Project Features

- Customer data analysis and visualization
- K-Means clustering
- Customer segment identification
- Interactive Streamlit dashboard
- Segment filtering
- Customer and segment summary statistics

## Project Structure

```text
customer-segmentation-system/
│
├── data/
│   └── Mall_Customers.csv
├── app.py
├── main.py
├── requirements.txt
└── .gitignore
```

## How to Run Locally

1. Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Project Goal

To apply unsupervised machine learning techniques to real-world customer data and build an interactive dashboard for customer segmentation.

---

**Author:** Adnan Maharoof

*An AI/ML learning project built using Python and Machine Learning.*
