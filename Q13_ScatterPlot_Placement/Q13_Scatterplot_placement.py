# pip install pandas seaborn matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('placement.csv')

# 🔍 Scatter Plot - CGPA vs Placement Exam Marks
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='cgpa', y='placement_exam_marks', hue='placed')
plt.title("Scatter Plot: CGPA vs Placement Exam Marks")
plt.xlabel("CGPA")
plt.ylabel("Placement Exam Marks")
plt.legend(title="Placed")
plt.grid(True)
plt.show()

# 🔍 Correlation Heatmap
plt.figure(figsize=(6, 5))
corr_matrix = df.corr(numeric_only=True)  # In case non-numeric columns exist
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
