# Fraud Detection Machine Learning Project

This repository contains code and analysis for a fraud detection machine learning project. In this project, we explore a dataset related to financial transactions and build machine learning models to detect fraudulent transactions.

dataset: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset

In this project, we aim to detect fraudulent transactions in a financial dataset using various machine learning algorithms. The primary goal is to build models that can identify fraudulent transactions accurately.

## Data
- The dataset contains information about financial transactions, including transaction type, amount, sender/receiver details, and more.
- It includes both legitimate and fraudulent transactions.
- The dataset is imbalanced, with a small percentage of fraudulent transactions.

## Analysis and Data Cleaning
- Imported necessary Python packages for data analysis and machine learning.
- Loaded the dataset and performed initial data analysis to understand its structure.
- Removed rows with missing values and converted the 'isFraud' column to integer type.

## Exploratory Data Analysis (EDA)
- Checked for duplicate values in the dataset.
- Visualized the distribution of transaction amounts to identify outliers.
- Analyzed the distribution of transaction types and their relationship with fraud.
- Explored correlations between various features in the dataset.

## Feature Engineering
- Preprocessed the data by capping and flooring outlier values in transaction amounts.
- Transformed categorical data (transaction types) into numerical format.
- Balanced the dataset using undersampling to address the class imbalance issue.

## Machine Learning Models
- Trained several machine learning models, including Logistic Regression, Support Vector Machine (SVM), Naive Bayes, K-Nearest Neighbors (KNN), and Decision Tree.
- Performed hyperparameter tuning using techniques like GridSearchCV and RandomizedSearchCV.
- Evaluated model performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

## Results
- The best-performing model achieved an accuracy of approximately 88% on the undersampled dataset.
- The ROC-AUC score and other metrics were also considered to assess model performance.

## Conclusion
- The project demonstrates the process of building machine learning models for fraud detection.
- It highlights the challenges of dealing with imbalanced datasets and the need for preprocessing techniques.
- The best-performing model can be further fine-tuned and deployed in a real-world fraud detection system.

Feel free to explore the code and analysis provided in this repository to gain insights into fraud detection using machine learning.
