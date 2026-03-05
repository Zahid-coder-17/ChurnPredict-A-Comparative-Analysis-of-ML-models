# Customer Churn Predictor 🔮

This repository contains a comparative analysis of different Machine Learning models to predict customer churn, along with a deployment-ready Streamlit application.

## 🚀 Live Demo
You can run the web app locally with Streamlit.
```bash
streamlit run app.py
```

## 📊 Model Performance
Benchmarked 5 ML models (XGBoost, SVM, Random Forest, Logistic Regression, KNN) on 7032-record dataset — best model: 59.1% F1  ·  83.1% ROC-AUC

## 📁 Repository Structure
- `data.csv`: The dataset used for training and testing.
- `train.py`: Script to train and benchmark the ML models.
- `app.py`: Streamlit frontend for interacting with the trained model.
- `churn_model.pkl`: Serialized best-performing model (Logistic Regression) + preprocessors.
- `metrics.txt`: Summarized model metric results.
- `page1.ipynb`: Original Jupyter Notebook analysis.

## Setup & Execution
1. Install requirements:
   ```bash
   pip install pandas scikit-learn xgboost streamlit joblib
   ```
2. Train the model (optional, as `churn_model.pkl` is pre-trained):
   ```bash
   python train.py
   ```
3. Run the interface:
   ```bash
   streamlit run app.py
   ```
