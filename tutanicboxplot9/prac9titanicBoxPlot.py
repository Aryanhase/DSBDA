import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset('titanic')
dataset.head()

# Set the figure size
plt.figure(figsize=(12, 8))

# Box plot
sns.boxplot(x='sex',y='age', data = dataset) 

sns.boxplot(x='sex', y='age', hue='survived', data=dataset)

# Title and labels
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')

# Show the plot
plt.show()
