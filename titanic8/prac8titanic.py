import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

dataset = sns.load_dataset('titanic')
dataset.head()

#distribution plots
sns.displot(x=dataset['age'], bins=10) #Visualizes the distribution of a single numerical variable using a histogram

sns.distplot(dataset['age'], bins = 10,kde=False) #KDE (Kernel Density Estimate)

#Joint plot
sns.jointplot(x = dataset['age'], y = dataset['fare'], kind ='scatter') #Displays a bivariate plot (relationship between two variables) along with their marginal distributions
sns.jointplot(x = dataset['age'], y = dataset['fare'], kind = 'hex') 

# rug plot
sns.rugplot(dataset['fare']) #Displays a small tick mark for each data point along a single axis to visualize the density

#Categorical plots

# bar plot
sns.barplot(x='sex', y='age', data=dataset) #Visualizes the relationship between a categorical variable and a numerical variable. Shows the mean by default

sns.barplot(x='sex', y='age', data=dataset, estimator=np.std) #estimator: Function to aggregate data (e.g., np.mean, np.std)

# box plot
sns.boxplot(x='sex', y='age', data=dataset) #Displays the distribution of a numerical variable through quartiles. Shows outliers as individual points

sns.boxplot(x='sex', y='age', data=dataset, hue="survived") #hue: Adds a categorical variable for layered comparison.

# violin plot
sns.violinplot(x='sex', y='age', data=dataset) #Combines aspects of a box plot and a KDE plot to show the distribution of a numerical variable

sns.violinplot(x='sex', y='age', data=dataset, hue='survived') 

#Advanced Plots

#strip plot
sns.stripplot(x='sex', y='age', data=dataset, jitter=False) #Draws a scatter plot where one axis is a categorical variable. Adds jitter to avoid overlapping points

sns.stripplot(x='sex', y='age', data=dataset, jitter=True)

sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived')

#swarm plot
sns.swarmplot(x='sex', y='age', data=dataset) #Similar to stripplot but adjusts the positions of points to avoid overlap, making the distribution clearer

sns.swarmplot(x='sex', y='age', data=dataset, hue='survived')

#heatmap 

#drop non-numeric columns
numeric_data = dataset.select_dtypes(include=[np.number])
corr = numeric_data.corr()

#plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - Numeric Data Only')
plt.show()

#price of titket distribution using histogram
sns.histplot(dataset['fare'], kde=False, bins=10) 
