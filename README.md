# Employee Retention Prediction using Machine Learning

## 📌 Project Overview
This project predicts whether an employee (data scientist) is likely to look for a job change using machine learning techniques. The objective is to help organizations take proactive, data-driven retention decisions.

## 🎯 Problem Statement
Employee attrition leads to increased hiring costs, loss of skilled talent, and reduced productivity. This project aims to identify employees at risk of job change based on demographic, educational, and professional attributes.

## 📊 Dataset Description
- Each row represents one employee
- Features include experience, education, company details, and training information
- Target variable:
  - `1` → Actively looking for a job change
  - `0` → Staying with the company

## ⚠️ Key Challenge
The dataset is **imbalanced**, with significantly fewer employees actively looking for a job change. Therefore, accuracy alone is misleading.

## 🧠 Machine Learning Workflow
- Data inspection and exploratory data analysis (EDA)
- Target imbalance analysis
- Feature–target split
- Stratified train–validation split
- Model training and evaluation
- Hyperparameter tuning
- SMOTE experimentation
- Final model selection

## 🤖 Models Implemented
- Logistic Regression
- Random Forest
- XGBoost (Final Model)
- LightGBM

## 📈 Evaluation Metrics
Due to class imbalance, the following metrics were used:
- Precision
- Recall
- **F1-score (Primary Metric)**
- ROC-AUC

## 🏆 Final Model
**Tuned XGBoost (without SMOTE)**  
Selected because it provided the best balance between precision and recall while avoiding excessive false positives.

## 🔧 Hyperparameter Tuning
- Method: RandomizedSearchCV
- Optimized Metric: F1-score
- Tuned parameters include tree depth, learning rate, number of estimators, sampling, and regularization parameters.

## 🔄 SMOTE Analysis
SMOTE was applied only on training data to handle class imbalance.
- Recall increased
- Precision and F1-score decreased  
Hence, the non-SMOTE tuned XGBoost model was retained.

## 🚀 Application
A Streamlit-based application was developed to predict job-change likelihood for new, unseen employee data.  
The application was tested locally, and the complete application code is included in this repository.

## 📁 Repository Structure

Employee-Retention-Prediction/
├── data/
├── notebooks/
├── model/
├── app/
├── main_xgb.py

