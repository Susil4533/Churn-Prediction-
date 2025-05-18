# Customer Churn Prediction Project

## Overview

This project aims to predict customer churn for a telecom company using machine learning and deep learning models. Accurately identifying customers likely to churn enables the company to take proactive retention actions, improving customer loyalty and revenue.

---

## Dataset

- **Source:** Telco Customer Churn dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Description:** Contains customer demographic information, account details, and service usage.
- **Target Variable:** `Churn` (1 = customer churned, 0 = customer retained)

---

## Data Preprocessing

- Removed unique identifiers (`customerID`)
- Converted `Churn` to binary numeric format
- Encoded categorical variables using label encoding
- Converted `TotalCharges` from object to float and handled missing values by dropping rows
- Scaled continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using StandardScaler

---

## Models Used

### 1. Logistic Regression

- A simple linear model serving as a strong baseline for binary classification.
- Achieved **81% accuracy** and **ROC-AUC of 0.84**.
- Strengths: Interpretability and good overall performance.

### 2. Random Forest

- An ensemble tree-based model that captures nonlinear relationships.
- Achieved **78% accuracy** and **ROC-AUC of 0.81**.
- Strengths: Robustness and ability to handle feature interactions.

### 3. TabTransformer

- A deep learning model based on transformer architecture adapted for tabular data.
- Achieved **78% accuracy** and **ROC-AUC of 0.808**.
- Strengths: Potential to capture complex feature interactions and dependencies.

---

## Evaluation Metrics

| Metric              | Logistic Regression | Random Forest | TabTransformer |
|---------------------|---------------------|---------------|----------------|
| Accuracy            | 0.81                | 0.78          | 0.78           |
| Precision (Churn)   | 0.66                | 0.62          | *0.58*         |
| Recall (Churn)      | 0.57                | 0.48          | *0.57*         |
| F1-score (Churn)    | 0.61                | 0.54          | *0.58*         |
| ROC-AUC             | 0.84                | 0.81          | *0.808*        |

---
**Why did the Transformer model underperform?**

- Transformers need large datasets to work well; our churn dataset is relatively small.
- Tabular data has simple structure and less complex feature interactions, which traditional models handle better.
- Transformers are prone to overfitting on small, structured datasets like this.

*Traditional models like Logistic Regression and Random Forest are more suitable for this type of data.*

---
## Summary

- The Logistic Regression model performed best overall, especially in identifying customers who churn (higher recall and ROC-AUC).
- Random Forest showed competitive performance but with slightly lower recall for churners.
- The TabTransformer model achieved comparable accuracy but requires further tuning to improve recall and precision.
- All models demonstrated challenges in predicting churners accurately due to class imbalance.

---

## Conclusion

This project successfully built and compared three models for customer churn prediction. While Logistic Regression currently leads in performance and interpretability, the TabTransformer model shows promise for capturing complex relationships in tabular data with further optimization.

The ability to predict churn provides the company with critical insights into which customers are at risk, enabling targeted retention strategies.

---

## Recommendations

- **Leverage the Logistic Regression model for immediate deployment** due to its strong performance and simplicity.
- **Further tune and optimize the TabTransformer model** to potentially surpass traditional models.
- **Address class imbalance** using techniques such as oversampling, class weights, or threshold tuning to improve recall on churners.
- **Integrate churn prediction with business processes**:
  - Use model predictions to prioritize outreach and personalized retention offers.
  - Monitor model performance regularly and retrain with fresh data.
- **Explore ensemble methods** combining strengths of multiple models for better accuracy and robustness.

---

## How to Run

1. Clone the repository.
2. Install dependencies:
