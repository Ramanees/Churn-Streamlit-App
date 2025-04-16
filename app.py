import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Dashboard layout
st.sidebar.title("Customer Churn Dashboard")
st.sidebar.markdown("Upload datasets and explore churn insights.")

# File uploaders
uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type=["csv"], key="main_data")
feedback_file = st.sidebar.file_uploader("Upload feedback data (CSV)", type=["csv"], key="feedback_data")

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

# Function to preprocess the uploaded data
def preprocess_data(df, global_sentiment=None):
    df_processed = df.copy()
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
    
    # Define expected features to match the model's training order
    expected_features = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain',
        'Gender_Male', 'Sentiment_Polarity'
    ]
    if global_sentiment is not None:
        df_processed["Sentiment_Polarity"] = global_sentiment
    else:
        df_processed["Sentiment_Polarity"] = 0  # Default if sentiment is missing
    # Add missing columns with default value 0
    for col in expected_features:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Add global sentiment if available
   
    
    # Reorder to expected features
    df_processed = df_processed[expected_features]
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
    
    # Ensure binary columns
    binary_cols = ["HasCrCard", "IsActiveMember", "Geography_Germany", "Geography_Spain", "Gender_Male"]
    for col in binary_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int)
    
    return df_processed, {}

# Function to analyze sentiment from "feedback" column
def analyze_sentiment(feedback_df):
    if feedback_df is not None and "feedback" in feedback_df.columns:
        feedback_df["Sentiment_Polarity"] = feedback_df["feedback"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        global_sentiment = feedback_df["Sentiment_Polarity"].mean()
        if "churn" in feedback_df.columns:
            global_churn = feedback_df["churn"].mean()
            return global_sentiment, global_churn
        return global_sentiment, None
    return None, None

# Function to predict churn
def predict_churn(model, data):
    try:
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]
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

# Main dashboard content
if uploaded_file is not None and model is not None:
    try:
        # Try different encodings for uploaded_file
        encoding_tried = False
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            encoding_tried = True
            try:
                df = pd.read_csv(uploaded_file, encoding='Windows-1252')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        st.markdown("### Data Preview")
        st.dataframe(df.head())
        
        global_sentiment = None
        global_churn = None
        if feedback_file is not None:
            try:
                feedback_df = pd.read_csv(feedback_file)
            except UnicodeDecodeError:
                try:
                    feedback_df = pd.read_csv(feedback_file, encoding='Windows-1252')
                except UnicodeDecodeError:
                    feedback_df = pd.read_csv(feedback_file, encoding='latin-1')
            
            st.markdown("### Feedback Data Preview")
            st.dataframe(feedback_df.head())
            global_sentiment, global_churn = analyze_sentiment(feedback_df)
            if global_sentiment is not None:
                df["Feedback_Churn"] = global_churn if global_churn is not None else -1
        
        df_processed, _ = preprocess_data(df, global_sentiment)
        predictions, probabilities = predict_churn(model, df_processed)
        
        if predictions is not None:
            df["Churn Prediction"] = predictions
            df["Churn Probability"] = probabilities
            df["Churn Prediction"] = df["Churn Prediction"].map({0: "Stay", 1: "Churn"})
            df = create_age_groups(df)
            
            st.markdown("### Prediction Results")
            all_columns = ["Churn Prediction", "Churn Probability", "Feedback_Churn"] + [col for col in df.columns if col not in ["Churn Prediction", "Churn Probability", "Feedback_Churn"]]
            st.dataframe(df[all_columns])
            
            st.download_button(
                label="Download Results as CSV",
                data=df.to_csv(index=False),
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
            
            # Dashboard Sections
            st.markdown("### Visual Insights Dashboard")
            
            # Section 1: Overall Churn
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
            
            # Section 2: Churn by Categories
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
            
            # Section 3: Churn vs. Activity and Sentiment
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
                    if "Sentiment_Polarity" in df.columns:
                        fig, ax = plt.subplots()
                        sns.boxplot(x="Churn Prediction", y="Sentiment_Polarity", data=df, ax=ax)
                        ax.set_title("Churn by Sentiment (Box)")
                        st.pyplot(fig)
            
            # Section 4: Advanced Plots
            with st.expander("Advanced Visualizations"):
                col1, col2 = st.columns(2)
                with col1:
                    if "Churn Probability" in df.columns:
                        fig, ax = plt.subplots()
                        sns.violinplot(x="Churn Prediction", y="Churn Probability", data=df, ax=ax)
                        ax.set_title("Churn Probability (Violin)")
                        st.pyplot(fig)
                with col2:
                    if "Sentiment_Polarity" in df.columns:
                        fig, ax = plt.subplots()
                        sns.histplot(df["Sentiment_Polarity"], bins=20, kde=True, ax=ax)
                        ax.set_title("Sentiment Distribution")
                        st.pyplot(fig)
            
            # Section 5: High-Risk Customers
            with st.expander("High-Risk Customers"):
                high_risk = df[df["Churn Probability"] > 0.5]
                if not high_risk.empty:
                    st.markdown("Customers with Churn Probability > 0.5:")
                    st.dataframe(high_risk[["CustomerId", "Churn Prediction", "Churn Probability", "Geography", "Age", "NumOfProducts", "IsActiveMember", "Sentiment_Polarity", "Feedback_Churn"]])
                else:
                    st.markdown("No high-risk customers found.")
            
            # Section 6: Prediction Accuracy (if feedback churn data exists)
            if "Feedback_Churn" in df.columns and -1 not in df["Feedback_Churn"].values:
                with st.expander("Prediction Accuracy"):
                    from sklearn.metrics import accuracy_score
                    actual_churn = df["Feedback_Churn"].astype(int)
                    predicted_churn = df["Churn Prediction"].map({"Stay": 0, "Churn": 1})
                    acc = accuracy_score(actual_churn, predicted_churn)
                    st.markdown(f"Accuracy compared to feedback churn: {acc:.2f}")
            
            # Section 7: Feature Importance
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
    st.sidebar.info("Please upload customer data and feedback data to start the analysis.")
