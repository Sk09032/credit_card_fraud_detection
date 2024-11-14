# Credit Card Fraud Detection using Machine Learning in Python

This project aims to predict whether a credit card transaction is fraudulent or not using machine learning techniques.

---

## Overview

The dataset contains transactions made by credit cardholders in Europe during September 2013. The dataset spans two days and includes 284,807 transactions, of which 492 are fraudulent. Due to the nature of the data, it is **highly imbalanced**, with the positive class (frauds) accounting for only **0.172%** of all transactions.

For confidentiality reasons, the input variables have been transformed into numerical values using **PCA (Principal Component Analysis)**. 

### Dataset Details:
- **Number of Transactions:** 284,807  
- **Number of Fraudulent Transactions:** 492  
- **Fraud Class Percentage:** 0.172%  

The dataset is available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).

---

## Project Goals

The main objective of this project is to:
- Accurately predict fraudulent transactions.
- Handle the class imbalance in the dataset.
- Evaluate the performance of various machine learning models.

---

## Technologies Used

- **Python**  
- **Pandas** and **NumPy** for data manipulation.  
- **Matplotlib** and **Seaborn** for data visualization.  
- **Scikit-learn** for machine learning algorithms and evaluation metrics.  
- **Imbalanced-learn** for handling imbalanced datasets.

---

## Insights from Model Results

Multiple machine learning models were implemented and evaluated to predict credit card fraud. Below are the results and insights from each model:

### 1. **Logistic Regression**
- **ROC_AUC Score:** 0.9660  
- **Performance Metrics:**  
  - Precision: 0.99 (Class 0), 0.91 (Class 1)  
  - Recall: 0.96 (Class 0), 0.97 (Class 1)  
  - F1-Score: 0.97 (Class 0), 0.94 (Class 1)  
- **Accuracy:** 96%  

**Insight:** Logistic Regression provides strong overall performance with high recall for fraud detection (Class 1), ensuring that most fraudulent transactions are captured.

---

### 2. **Support Vector Classifier (SVC)**
- **ROC_AUC Score:** 0.9732  
- **Performance Metrics:**  
  - Precision: 0.99 (Class 0), 0.91 (Class 1)  
  - Recall: 0.96 (Class 0), 0.99 (Class 1)  
  - F1-Score: 0.98 (Class 0), 0.95 (Class 1)  
- **Accuracy:** 97%  

**Insight:** SVC achieved the highest accuracy among the models tested, excelling in recall for fraudulent transactions, making it reliable for minimizing false negatives.

---

### 3. **Decision Tree Classifier**
- **ROC_AUC Score:** 0.9573  
- **Performance Metrics:**  
  - Precision: 0.98 (Class 0), 0.91 (Class 1)  
  - Recall: 0.96 (Class 0), 0.96 (Class 1)  
  - F1-Score: 0.97 (Class 0), 0.93 (Class 1)  
- **Accuracy:** 96%  

**Insight:** While the Decision Tree Classifier performed well, its metrics slightly lag behind those of SVC and KNeighborsClassifier. It may benefit from hyperparameter tuning or ensemble methods.

---

### 4. **Random Forest Classifier**
- **ROC_AUC Score:** 0.9724  
- **Performance Metrics:**  
  - Precision: 1.00 (Class 0), 0.89 (Class 1)  
  - Recall: 0.95 (Class 0), 1.00 (Class 1)  
  - F1-Score: 0.97 (Class 0), 0.94 (Class 1)  
- **Accuracy:** 96%  

**Insight:** Random Forest achieved perfect recall for fraud detection (Class 1), ensuring that all fraudulent transactions were detected. Precision for fraud detection, however, could be slightly improved.

---

### 5. **K-Nearest Neighbors (KNeighborsClassifier)**
- **ROC_AUC Score:** 0.9803  
- **Performance Metrics:**  
  - Precision: 0.98 (Class 0), 0.99 (Class 1)  
  - Recall: 1.00 (Class 0), 0.96 (Class 1)  
  - F1-Score: 0.99 (Class 0), 0.98 (Class 1)  
- **Accuracy:** 99%  

**Insight:** KNeighborsClassifier outperformed all other models with the highest ROC_AUC score and accuracy. Its balanced precision, recall, and F1-score make it the best-performing model for both fraud detection and non-fraud detection.

---

## Recommendations
Based on the results:
- **KNeighborsClassifier** and **SVC** are the most effective models for this dataset, given their high accuracy and strong ROC_AUC scores.  
- For real-world applications where recall for fraudulent transactions is critical, **Random Forest** could also be a reliable choice due to its perfect recall for Class 1.
- Further optimization (e.g., hyperparameter tuning, ensemble methods) can be explored for improving performance further.

---  

[LinkedIn](https://linkedin.com/in/sunny-kumar-0b02ba204).

---
