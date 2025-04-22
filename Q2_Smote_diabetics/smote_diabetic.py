# Q2
# Use SMOTE technique to generate synthetic data on diabetic dataset.Comment on the F1-score values before and after SMOTE
# pip install pandas scikit-learn imbalanced-learn

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

# Step 2: Load the dataset
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Step 3: Split into features and target
X = df.drop('Outcome', axis=1)  # Assuming 'Outcome' is the target
y = df['Outcome']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 5: Train model BEFORE SMOTE
model_before = RandomForestClassifier(random_state=42)
model_before.fit(X_train, y_train)
y_pred_before = model_before.predict(X_test)

# Step 6: Evaluate F1-score BEFORE SMOTE
f1_before = f1_score(y_test, y_pred_before)
print("F1-Score BEFORE SMOTE:", round(f1_before, 4))
print("\nClassification Report BEFORE SMOTE:\n", classification_report(y_test, y_pred_before))

# Step 7: Apply SMOTE on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 8: Train model AFTER SMOTE
model_after = RandomForestClassifier(random_state=42)
model_after.fit(X_train_smote, y_train_smote)
y_pred_after = model_after.predict(X_test)

# Step 9: Evaluate F1-score AFTER SMOTE
f1_after = f1_score(y_test, y_pred_after)
print("F1-Score AFTER SMOTE:", round(f1_after, 4))
print("\nClassification Report AFTER SMOTE:\n", classification_report(y_test, y_pred_after))
