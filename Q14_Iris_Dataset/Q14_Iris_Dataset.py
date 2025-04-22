# pip install pandas seaborn matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset("iris")

# Display the first few rows (optional)
print(iris.head())

# Scatter plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species', style='species', palette='deep')

# Add plot title and labels
plt.title('Sepal Length vs Petal Length in Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.legend(title='Species')
plt.show()
