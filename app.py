import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from io import StringIO

# Set page title
st.title("Customer Churn Prediction App")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=["csv"])

# Load the pre-trained model
@st.cache_resource
def load_model(model_path="churn_model.pkl"):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'churn_model.pkl' is in the directory.")
        return None

model = load_model()

# Function to preprocess the uploaded data to match notebook's pipeline
def preprocess_data(df):
    df_processed = df.copy()
    # Drop columns only if they exist
    columns_to_drop = ["RowNumber", "CustomerId", "Surname", "Exited"]
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors='ignore')
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype in ["int64", "float64"]:
            df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    # One-hot encode categorical variables
    categorical_cols = ["Geography", "Gender"]
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=[col], prefix=[col], drop_first=True)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
    
    # Ensure binary columns (HasCrCard, IsActiveMember) are kept as is
    binary_cols = ["HasCrCard", "IsActiveMember"]
    for col in binary_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int)
    
    # Define expected features based on one-hot encoding
    expected_features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
        "Geography_Spain", "Geography_Germany", "Gender_Male", "HasCrCard", "IsActiveMember"
    ]  # 11 features; adjust to 12 with exact notebook output
    if len(expected_features) < 12:
        expected_features.append("Extra_Feature")  # Replace with actual feature from notebook
    
    # Reorder and add missing columns with zeros
    missing_cols = [col for col in expected_features if col not in df_processed.columns]
    for col in missing_cols:
        df_processed[col] = 0
    df_processed = df_processed[expected_features]
    
    # Debug: Check the shape
    st.write("Processed feature shape:", df_processed.shape)
    
    return df_processed, {}

# Function to predict churn and get probabilities
def predict_churn(model, data):
    try:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]  # Probability of churn (class 1)
        return predictions, probabilities
    except AttributeError:
        st.error("Model does not support predictions.")
        return None, None

# Function to create age groups
def create_age_groups(df):
    bins = [0, 30, 40, 50, 60, 100]
    labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

if uploaded_file is not None and model is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        
        # Preprocess the data
        df_processed, _ = preprocess_data(df)
        
        # Predict churn
        predictions, probabilities = predict_churn(model, df_processed)
        
        if predictions is not None:
            # Add predictions and probabilities to the original dataframe
            df["Churn Prediction"] = predictions
            df["Churn Probability"] = probabilities
            
            # Map predictions to labels
            df["Churn Prediction"] = df["Churn Prediction"].map({0: "Stay", 1: "Churn"})
            
            # Create age groups
            df = create_age_groups(df)
            
            # Display results (avoiding duplicate columns)
            st.subheader("Prediction Results")
            # Select all columns, ensuring no duplicates by using a set or unique list
            all_columns = ["Churn Prediction", "Churn Probability"] + [col for col in df.columns if col not in ["Churn Prediction", "Churn Probability"]]
            st.dataframe(df[all_columns])
            
            # Download button for results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
            
            # Visual Insights
            st.subheader("Visual Insights")
            
            # 1. Overall Churn Prediction (Bar Chart)
            st.write("Overall Churn Prediction (Bar Chart)")
            fig, ax = plt.subplots()
            sns.countplot(x="Churn Prediction", data=df, ax=ax)
            ax.set_xlabel("Prediction")
            ax.set_ylabel("Count")
            ax.set_title("Churn vs. Stay")
            st.pyplot(fig)
            
            # 2. Overall Churn Prediction (Pie Chart)
            st.write("Overall Churn Prediction (Pie Chart)")
            fig, ax = plt.subplots()
            df["Churn Prediction"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_title("Churn vs. Stay Percentage")
            st.pyplot(fig)
            
            # 3. Churn Probability Histogram
            st.write("Churn Probability Histogram")
            fig, ax = plt.subplots()
            sns.histplot(probabilities, bins=20, kde=True, ax=ax)
            ax.set_xlabel("Churn Probability")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Churn Probabilities")
            st.pyplot(fig)
            
            # 4. Churn by Geography
            st.write("Churn by Geography")
            if "Geography" in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(x="Geography", hue="Churn Prediction", data=df, ax=ax)
                ax.set_xlabel("Geography")
                ax.set_ylabel("Count")
                ax.set_title("Churn by Geography")
                st.pyplot(fig)
            
            # 5. Churn by Age Group
            st.write("Churn by Age Group")
            if "AgeGroup" in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(x="AgeGroup", hue="Churn Prediction", data=df, ax=ax)
                ax.set_xlabel("Age Group")
                ax.set_ylabel("Count")
                ax.set_title("Churn by Age Group")
                st.pyplot(fig)
            
            # 6. Churn by Number of Products
            st.write("Churn by Number of Products")
            if "NumOfProducts" in df.columns:
                fig, ax = plt.subplots()
                sns.countplot(x="NumOfProducts", hue="Churn Prediction", data=df, ax=ax)
                ax.set_xlabel("Number of Products")
                ax.set_ylabel("Count")
                ax.set_title("Churn by Number of Products")
                st.pyplot(fig)
            
            # 7. Churn vs. Activity
            st.write("Churn vs. Activity")
            if "IsActiveMember" in df.columns:
                df["IsActiveMember"] = df["IsActiveMember"].map({0: "Inactive", 1: "Active"})
                fig, ax = plt.subplots()
                sns.countplot(x="IsActiveMember", hue="Churn Prediction", data=df, ax=ax)
                ax.set_xlabel("Activity Status")
                ax.set_ylabel("Count")
                ax.set_title("Churn vs. Activity")
                st.pyplot(fig)
            
            # 8. Highlight High-Risk Customers
            st.subheader("High-Risk Customers")
            high_risk = df[df["Churn Probability"] > 0.5]
            if not high_risk.empty:
                st.write("Customers with Churn Probability > 0.5:")
                st.dataframe(high_risk[["CustomerId", "Churn Prediction", "Churn Probability"] + ["Geography", "Age", "NumOfProducts", "IsActiveMember"]])
            else:
                st.write("No high-risk customers found.")
            
            # 9. Feature Importance (XGBoost specific)
            try:
                feature_importance = model.feature_importances_
                feature_names = df_processed.columns
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": feature_importance
                }).sort_values(by="Importance", ascending=False)
                
                st.write("Feature Importance")
                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=importance_df.head(10), ax=ax)
                ax.set_title("Top 10 Features for Churn Prediction")
                st.pyplot(fig)
            except AttributeError:
                st.error("Feature importance not available")
                
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
else:
    st.info("Please upload a CSV file to predict churn.")