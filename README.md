# 📊 Customer Churn Prediction Dashboard with Sentiment Analysis

An interactive **Streamlit dashboard** that predicts customer churn using a machine learning model, enhanced with **sentiment analysis** from customer feedback.

🔗 **Live App**: [churn-analysis-dashboard.onrender.com](https://churn-analysis-dashboard.onrender.com/)

---

## 🧠 Features

✅ Upload customer dataset (CSV format)  
✅ Predict churn vs. stay using a trained model  
✅ Analyze sentiment from feedback using **TextBlob**  
✅ Explore interactive visualizations  
✅ View and download high-risk customers  
✅ Export results as downloadable CSV  

---

## 📸 Interface Preview

### 🔍 Overview & Churn Analysis

![Overall Churn Analysis](https://github.com/Ramanees/Churn-Streamlit-App/blob/main/screenshots/Screenshot%20(35).png)

### 📊 By Category & Sentiment

![Churn Categories](https://github.com/Ramanees/Churn-Streamlit-App/blob/main/screenshots/Screenshot%20(36).png)

### ⚠️ High-Risk Customers

![High Risk](https://github.com/Ramanees/Churn-Streamlit-App/blob/main/screenshots/Screenshot%20(38).png)


---

## 📥 How to Use

### 1. Upload Dataset

- Click `Browse files` or drag and drop your `.csv` file on the sidebar.
- Required column: `feedback` (text)  
- Recommended columns:
  - `CustomerId`, `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`
  - `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`

### 2. View Results

- A table of predictions will show:
  - `Churn Prediction` (Stay/Churn)
  - `Churn Probability`
  - `Sentiment Polarity` of feedback

### 3. Explore Visuals

- Bar chart & pie chart of churn status  
- Churn by geography, age group, products  
- Churn vs. activity & sentiment  
- Violin/box plots and histograms  
- Top 10 most important features (if model supports)

### 4. Download CSV

- Click **Download Results as CSV** to save predictions locally  
  - File: `churn_predictions_with_feedback.csv`

---

