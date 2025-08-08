import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from collections import Counter

# Data Preprocessing Function

def preprocess_data(df1):
    num_cols = df1.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df1[col].fillna(df1[col].median(), inplace=True)

    cat_cols = df1.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df1[col].fillna(df1[col].mode()[0], inplace=True)
        df1[col] = le.fit_transform(df1[col])

    df1['session_date'] = pd.to_datetime(df1[['year', 'month', 'day']])
    df1['session_duration'] = df1.groupby('session_id')['page'].transform('count')
    df1['avg_price'] = (df1['price'] + df1['price_2']) / 2

    df1.drop(columns=['year', 'month', 'day', 'session_date', 'session_id'], inplace=True, errors='ignore')
    return df1

# Classification

def train_classification_model(df1):
    df1 = preprocess_data(df1)
    X = df1.drop(columns=['order'])
    y = df1['order']

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    class_counts = Counter(y)
    min_class_size = min(class_counts.values())

    if min_class_size < 2:
        X_resampled, y_resampled = X, y
    else:
        k_neighbors = min(5, min_class_size - 1)
        smote = SMOTE(k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return model, metrics

# Regression

def train_regression_model(df1):
    df1 = preprocess_data(df1)
    df1['revenue'] = df1['price'] * df1['order']
    X = df1.drop(columns=['revenue'])
    y = df1['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R-squared': r2_score(y_test, y_pred)
    }
    return model, metrics


# Clustering

def train_clustering_model(df1):
    df1 = preprocess_data(df1)
    X = df1.drop(columns=['order'], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    metrics = {
        'Silhouette Score': silhouette_score(X_scaled, labels),
        'Davies-Bouldin Index': davies_bouldin_score(X_scaled, labels)
    }

    df1['Cluster'] = labels
    return df1[['Cluster']], metrics


# Streamlit Interface

st.set_page_config(page_title="Customer Conversion App", layout="centered")
st.title("🛒 Customer Conversion Analysis using Clickstream Data")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df1 = pd.read_csv(uploaded_file)
    st.success("✅ File Uploaded Successfully!")

    task = st.selectbox("📌 Choose a Task", ["Classification", "Regression", "Clustering"])

    if task == "Classification":
        if st.button("Predict"):
            model, metrics = train_classification_model(df1)
            st.subheader("📈 Classification Results")
            for k, v in metrics.items():
                st.write(f"**{k}**: {v:.2f}")

    elif task == "Regression":
        if st.button("Predict"):
            model, metrics = train_regression_model(df1)
            st.subheader("📉 Regression Results")
            for k, v in metrics.items():
                st.write(f"**{k}**: {v:.2f}")

    elif task == "Clustering":
        if st.button("Predict"):
            clustered_df, metrics = train_clustering_model(df1)
            st.subheader("📊 Clustering Results")
            st.dataframe(clustered_df.head())
            for k, v in metrics.items():
                st.write(f"**{k}**: {v:.2f}")

   