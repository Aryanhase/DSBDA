# Import the required libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
dataset = sns.load_dataset('iris')

# Show the first 5 rows of the dataset
print(dataset.head())

#   HISTOGRAM PLOTS

# Create 2 rows and 2 columns of empty plots (figures)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # Size of the whole plot is 12x8

# Plot a histogram for sepal length
sns.histplot(dataset['sepal_length'], ax=axes[0, 0], kde=True, color='skyblue')
axes[0, 0].set_title('Sepal Length')

# Plot a histogram for sepal width
sns.histplot(dataset['sepal_width'], ax=axes[0, 1], kde=True, color='salmon')
axes[0, 1].set_title('Sepal Width')

# Plot a histogram for petal length
sns.histplot(dataset['petal_length'], ax=axes[1, 0], kde=True, color='limegreen')
axes[1, 0].set_title('Petal Length')

# Plot a histogram for petal width
sns.histplot(dataset['petal_width'], ax=axes[1, 1], kde=True, color='violet')
axes[1, 1].set_title('Petal Width')

# Adjust layout so plots don't overlap
plt.tight_layout()

# Show all histogram plots
plt.show()

#   BOXPLOTS

# Create 2 rows and 2 columns of empty plots (figures) again for boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Draw a boxplot of sepal length for each species
sns.boxplot(y='sepal_length', x='species', data=dataset, ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length by Species')

# Draw a boxplot of sepal width for each species
sns.boxplot(y='sepal_width', x='species', data=dataset, ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width by Species')

# Draw a boxplot of petal length for each species
sns.boxplot(y='petal_length', x='species', data=dataset, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length by Species')

# Draw a boxplot of petal width for each species
sns.boxplot(y='petal_width', x='species', data=dataset, ax=axes[1, 1])
axes[1, 1].set_title('Petal Width by Species')

# Adjust layout so plots don't overlap
plt.tight_layout()

# Show all boxplots
plt.show()
