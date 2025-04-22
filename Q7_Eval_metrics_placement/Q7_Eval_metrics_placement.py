# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, precision_recall_curve)

# Load data
df = pd.read_csv('placement.csv')

# Split data
X = df[['cgpa', 'placement_exam_marks']]
y = df['placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr = lr.predict_proba(X_test)[:,1]

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]

def evaluate_classifier(y_true, y_pred, y_proba, model_name):
    """Calculate and display multiple classification metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_proba),
        'Confusion Matrix': confusion_matrix(y_true, y_pred)
    }
    
    print(f"\n{model_name} Performance Metrics:")
    print("--------------------------------")
    for name, value in metrics.items():
        if name != 'Confusion Matrix':
            print(f"{name}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("Confusion Matrix:")
    print(metrics['Confusion Matrix'])
    
    return metrics

# Evaluate both models
lr_metrics = evaluate_classifier(y_test, y_pred_lr, y_proba_lr, "Logistic Regression")
rf_metrics = evaluate_classifier(y_test, y_pred_rf, y_proba_rf, "Random Forest")

# For Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Random Forest Feature Importance')
plt.show()

# For Logistic Regression
coefficients = pd.DataFrame({
    'Feature': ['Intercept'] + list(X.columns),
    'Coefficient': [lr.intercept_[0]] + list(lr.coef_[0])
})

print("\nLogistic Regression Coefficients:")
print(coefficients)







# OR


#  Implement and explore performance evaluation metrics for placement dataset.

# # 1. Import Libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
# )

# # 2. Load Dataset
# file_path = 'placement.csv'
# df = pd.read_csv(file_path)
# print("First 5 rows:\n", df.head())

# # 3. Preprocess
# print("\nColumn types:\n", df.dtypes)

# # Encoding categorical features if any
# for col in df.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])

# # 4. Feature-Target Split
# # Assuming last column is 'status' or 'placed'
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# # 5. Feature Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 6. Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# # 7. Model Training
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# # 8. Evaluation Metrics
# print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"\nAccuracy:  {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall:    {recall:.2f}")
# print(f"F1-Score:  {f1:.2f}")

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # ROC Curve
# y_proba = model.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_proba)
# roc_auc = roc_auc_score(y_test, y_proba)

# plt.figure(figsize=(8,5))
# plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
# plt.plot([0,1], [0,1], 'k--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC-AUC Curve")
# plt.legend()
# plt.grid(True)
# plt.show()
