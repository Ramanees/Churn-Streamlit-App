import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Dashboard layout
st.sidebar.title("Customer Churn Analysis Dashboard")
st.sidebar.markdown("Upload customer dataset with feedback.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type=["csv"], key="main_data")

# Load the pre-trained model
@st.cache_resource
def load_model(model_path="churn_model_with_sentiment.pkl"):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'churn_model_with_sentiment.pkl' is in the directory.")
        return None

model = load_model()

# Function to analyze sentiment and get average polarity
def analyze_sentiment_column(df):
    if "feedback" in df.columns:
        df["Sentiment_Polarity"] = df["feedback"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    else:
        df["Sentiment_Polarity"] = 0
    return df

# Preprocessing function
def preprocess_data(df):
    df_processed = df.copy()
    columns_to_drop = ["RowNumber", "CustomerId", "Surname", "Exited", "feedback"]
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors='ignore')

    for col in df_processed.columns:
        if df_processed[col].dtype in ["int64", "float64"]:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

    categorical_cols = ["Geography", "Gender"]
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=[col], prefix=[col], drop_first=True)

    expected_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain',
        'Gender_Male', 'Sentiment_Polarity'
    ]

    for col in expected_features:
        if col not in df_processed.columns:
            df_processed[col] = 0

    df_processed = df_processed[expected_features]

    scaler = StandardScaler()
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    for col in numeric_cols:
        df_processed[col] = scaler.fit_transform(df_processed[[col]])

    binary_cols = ["HasCrCard", "IsActiveMember", "Geography_Germany", "Geography_Spain", "Gender_Male"]
    for col in binary_cols:
        df_processed[col] = df_processed[col].astype(int)

    return df_processed

# Predict churn
def predict_churn(model, data):
    try:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
        return predictions, probabilities
    except AttributeError:
        st.error("Model does not support predictions.")
        return None, None

# Age groups for visualization
def create_age_groups(df):
    bins = [0, 30, 40, 50, 60, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

# Streamlit main logic
if uploaded_file is not None and model is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Data Preview")
        st.dataframe(df.head())

        df = analyze_sentiment_column(df)
        df_processed = preprocess_data(df)

        predictions, probabilities = predict_churn(model, df_processed)

        if predictions is not None:
            df["Churn Prediction"] = predictions
            df["Churn Probability"] = probabilities
            df["Churn Prediction"] = df["Churn Prediction"].map({0: "Stay", 1: "Churn"})
            df = create_age_groups(df)

            st.markdown("### Prediction Results")
            st.dataframe(df[["CustomerId", "Churn Prediction", "Churn Probability", "Sentiment_Polarity", "feedback"]])

            st.download_button(
                label="Download Results as CSV",
                data=df.to_csv(index=False),
                file_name="churn_predictions_with_feedback.csv",
                mime="text/csv"
            )

            # Dashboard Sections
            st.markdown("### Visual Insights Dashboard")

            # Churn Analysis
            with st.expander("Overall Churn Analysis"):
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    sns.countplot(x="Churn Prediction", data=df, ax=ax)
                    ax.set_title("Churn vs. Stay (Bar)")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots()
                    df["Churn Prediction"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                    ax.set_title("Churn vs. Stay (Pie)")
                    st.pyplot(fig)

            with st.expander("Churn by Categories"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "Geography" in df.columns:
                        fig, ax = plt.subplots()
                        sns.countplot(x="Geography", hue="Churn Prediction", data=df, ax=ax)
                        ax.set_title("Churn by Geography")
                        st.pyplot(fig)
                with col2:
                    if "AgeGroup" in df.columns:
                        fig, ax = plt.subplots()
                        sns.countplot(x="AgeGroup", hue="Churn Prediction", data=df, ax=ax)
                        ax.set_title("Churn by Age Group")
                        st.pyplot(fig)
                with col3:
                    if "NumOfProducts" in df.columns:
                        fig, ax = plt.subplots()
                        sns.countplot(x="NumOfProducts", hue="Churn Prediction", data=df, ax=ax)
                        ax.set_title("Churn by Products")
                        st.pyplot(fig)

            with st.expander("Churn vs. Activity & Sentiment"):
                col1, col2 = st.columns(2)
                with col1:
                    if "IsActiveMember" in df.columns:
                        df["IsActiveMember"] = df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
                        fig, ax = plt.subplots()
                        sns.countplot(x="IsActiveMember", hue="Churn Prediction", data=df, ax=ax)
                        ax.set_title("Churn vs. Activity")
                        st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots()
                    sns.boxplot(x="Churn Prediction", y="Sentiment_Polarity", data=df, ax=ax)
                    ax.set_title("Churn by Sentiment")
                    st.pyplot(fig)

            with st.expander("Advanced Visualizations"):
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots()
                    sns.violinplot(x="Churn Prediction", y="Churn Probability", data=df, ax=ax)
                    ax.set_title("Churn Probability (Violin)")
                    st.pyplot(fig)
                with col2:
                    fig, ax = plt.subplots()
                    sns.histplot(df["Sentiment_Polarity"], bins=20, kde=True, ax=ax)
                    ax.set_title("Sentiment Distribution")
                    st.pyplot(fig)

            with st.expander("High-Risk Customers"):
                high_risk = df[df["Churn Probability"] > 0.5]
                st.dataframe(high_risk[["CustomerId", "Churn Prediction", "Churn Probability", "Age", "NumOfProducts", "IsActiveMember", "Sentiment_Polarity"]])

            with st.expander("Feature Importance"):
                try:
                    feature_importance = model.feature_importances_
                    feature_names = df_processed.columns
                    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance}).sort_values(by="Importance", ascending=False)
                    fig, ax = plt.subplots()
                    sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), ax=ax)
                    ax.set_title("Top 10 Features")
                    st.pyplot(fig)
                except AttributeError:
                    st.error("Feature importance not available")

    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.sidebar.info("Please upload the customer dataset with feedback included.")
