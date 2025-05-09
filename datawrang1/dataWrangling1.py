import pandas as pd

dataFrame = pd.read_csv(r"n_movies.csv")
print(dataFrame.head(20))

dataFrame.head(n = 5) #return first n rows
dataFrame.tail(n = 6) #return last n rows

dataFrame.index #the index (row labels) of the dataset

dataFrame.columns #the column labels of the dataset

dataFrame.shape  #returns a tuple representing the dimensionality of the dataset

dataFrame.dtypes #returns the datatypes in the dataset 

dataFrame.columns.values #returns the col values in the dataset in array format

dataFrame.describe(include = 'all') #generate descriptive statistics
dataFrame['rating'].describe()

dataFrame['title'] #read the data col-wise

dataFrame.sort_index(axis = 1, ascending = False) #sort objs by label (along the axis)

dataFrame.sort_values(by = "year") #sort vals by col names

dataFrame.iloc[5] #purely integer-location based indexing for selection by position
dataFrame.iloc[0:3, [0, 8]]

dataFrame[0:3] #select via [], which slices the rows

dataFrame.loc[:, ["title", "year"]] #selection by label

dataFrame.iloc[:5, :] #a subset of first 5 rows of the original data

dataFrame.iloc[:, :4] #a subset of first 4 cols of the original data

dataFrame.iloc[:10, :2] #a subset of first 10 rows and 2 cols of the original data
