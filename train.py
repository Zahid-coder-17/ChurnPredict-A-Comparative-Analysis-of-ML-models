import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv('data.csv')
num_records = len(df)

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Drop customerID
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Separate X and y
X = df.drop('Churn', axis=1)
y = df['Churn']

# Label encode target
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier()
}

results = {}
evaluation_records = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {'Precision': precision, 'Recall': recall, 'F1': f1, 'ROC_AUC': roc_auc, 'model': model}
    evaluation_records.append({
        'Model': name,
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1 Score': f"{f1:.4f}",
        'ROC-AUC': f"{roc_auc:.4f}"
    })

# Evaluation Framework Output
eval_df = pd.DataFrame(evaluation_records)
print("\n--- Model Evaluation Framework Results ---")
print(eval_df.to_string(index=False))
print("------------------------------------------\n")

# Find best model based on F1
best_model_name = max(results, key=lambda k: results[k]['F1'])
best_model = results[best_model_name]['model']
best_f1 = results[best_model_name]['F1'] * 100
best_roc = results[best_model_name]['ROC_AUC'] * 100

summary_text = f"Benchmarked 5 ML models (XGBoost, SVM, Random Forest, Logistic Regression, KNN) on {num_records}-record dataset — best model: {best_f1:.1f}% F1  ·  {best_roc:.1f}% ROC-AUC"
print(summary_text)

# Save the best model and preprocessing objects
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'encoders': encoders,
    'categorical_cols': categorical_cols.tolist(),
    'features': X.columns.tolist()
}, 'churn_model.pkl')

with open('metrics.txt', 'w') as f:
    f.write(summary_text + "\n\n")
    f.write("--- Model Evaluation Framework Results ---\n")
    f.write(eval_df.to_string(index=False) + "\n")
    f.write("------------------------------------------\n")
