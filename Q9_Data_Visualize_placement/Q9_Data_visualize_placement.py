# Explore data visualization techniques on placement dataset.
# pip install pandas seaborn matplotlib


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('placement.csv')

# 1. 🔍 Count Plot - Placement Status
plt.figure(figsize=(6, 4))
sns.countplot(x='placed', data=df)
plt.title("Placement Status Distribution")
plt.xlabel("Placed")
plt.ylabel("Count")
plt.show()

# 2. 📉 Distribution Plot - CGPA
plt.figure(figsize=(6, 4))
sns.histplot(df['cgpa'], kde=True, bins=15)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.ylabel("Frequency")
plt.show()

# 3. 📊 Distribution Plot - Placement Exam Marks
plt.figure(figsize=(6, 4))
sns.histplot(df['placement_exam_marks'], kde=True, bins=15)
plt.title("Placement Exam Marks Distribution")
plt.xlabel("Placement Exam Marks")
plt.ylabel("Frequency")
plt.show()

# 4. 📈 Scatter Plot - CGPA vs Placement Exam Marks
plt.figure(figsize=(6, 4))
sns.scatterplot(x='cgpa', y='placement_exam_marks', hue='placed', data=df)
plt.title("CGPA vs Placement Exam Marks")
plt.xlabel("CGPA")
plt.ylabel("Placement Exam Marks")
plt.legend(title='Placed')
plt.show()
