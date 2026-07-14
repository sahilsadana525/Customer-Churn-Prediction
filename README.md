# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer is likely to churn based on behavioral, subscription, payment, and interaction data. The project uses a dataset containing over **440,000 customer records** and includes an interactive **Streamlit web application** for real-time churn risk assessment.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Predicting customers who are likely to leave can help companies take proactive retention measures.

This project analyzes customer attributes such as:

- Age and tenure
- Usage frequency
- Support calls
- Payment delays
- Subscription type
- Contract length
- Total spending
- Last interaction

The application takes customer information as input and provides a **churn prediction, model churn score, and risk level**.

## Machine Learning Workflow

The project follows an end-to-end ML workflow:

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Numerical feature scaling and categorical encoding
4. Training and comparison of multiple classification models
5. Model evaluation using various performance metrics
6. Feature importance and distribution-shift analysis
7. Deployment using Streamlit

## Models Evaluated

Three machine learning models were compared:

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 89.34% | 90.40% | 95.90% |
| Random Forest | 99.64% | 99.68% | ~100% |
| XGBoost | **99.99%** | **99.99%** | **100%** |

**XGBoost** achieved the best performance on the internal validation set and was selected as the final model for the web application.

## Distribution Shift Analysis

A significant performance difference was observed between the internal validation set and the separately provided external test dataset. Further analysis revealed distribution shifts in important features such as **Support Calls, Payment Delay, Total Spend, Contract Length, and Gender**.

This demonstrates an important real-world ML challenge: **excellent in-distribution validation performance does not always guarantee strong generalization when the underlying data distribution changes.**

## Web Application

The Streamlit application allows users to enter customer information and receive:

- Model churn score
- Churn prediction
- Risk classification
- Key risk indicators
- Customer retention recommendations

## Technology Stack

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Matplotlib

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── churn_preprocessor.pkl
├── churn_xgboost_model.json
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Key Learnings

This project demonstrates practical experience in building an end-to-end machine learning pipeline, comparing classification algorithms, analyzing feature importance, identifying data distribution shifts, and deploying a trained model through an interactive web application.

## Author

**Sahil Sadana**  
M.Tech AI & Data Science | Machine Learning Enthusiast
