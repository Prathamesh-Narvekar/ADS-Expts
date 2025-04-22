# pip install pandas numpy matplotlib seaborn scikit-learn

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Data Collection
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)
print("First 5 rows of dataset:\n", df.head())

# Step 3: Data Cleaning & Preprocessing
print("\nChecking for null values:\n", df.isnull().sum())

# Visualizing missing data (if any)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Scaling features
X = df.drop('Outcome', axis=1)
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Exploratory Data Analysis
print("\nClass distribution:\n", df['Outcome'].value_counts())

# Visualizing class distribution
sns.countplot(x='Outcome', data=df)
plt.title("Class Distribution")
plt.show()

# Visualize correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 5: Model Building
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=df.columns[:-1])
feat_importances.sort_values().plot(kind='barh', figsize=(10, 6), title="Feature Importance")
plt.show()
