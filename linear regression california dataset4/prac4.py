from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt  

#loading dataframe
california_housing = fetch_california_housing()
df = pd.DataFrame(california_housing.data)

df.columns = california_housing.feature_names #sets the column names to the provided feature names.
df['Price'] = california_housing.target #adds a new column Price containing the target variable values.
df

#preprocessing
df.isnull().sum()

#splitting data into features x and y(target)
X = df.drop(['Price'],axis=1) #all columns except the target cols 'Price'
y  = df['Price'] #only the target col

# train test split (splits the data into training and testing sets)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0 ) #20% of the data will be used for testing, and 80% for training
#random_state=0: Ensures reproducibility of the split

#training the model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, y_train) #trained model can now predict the target (price) based on other features

#making predictions on test set
y_test_pred = lm.predict(X_test) #generates predictions using the test feature set X_test.

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_test_pred) #measures the average squared difference between the predicted and actual values
print(mse)

rts = r2_score(y_test, y_test_pred) #how well the regression model fits the data (0-1)
print(rts) #r2_score

#plotting the actual vs predicted values
plt.figure(figsize=(12, 6)) #creates a new figure with a specific size (12x6 inches), allowing more space for visual clarity

#plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue') #Plots the actual prices (y_test) vs predicted prices (y_test_pred) same color

# Actual Prices with Circle Marker
plt.scatter(y_test, y_test, alpha=0.7, color='blue', label='Actual Price')

# Predicted Prices with Cross Marker
plt.scatter(y_test, y_test_pred, alpha=0.7, color='red', label='Predicted Price')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2) #line represents the ideal prediction line where actual values equal predicted values
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.grid(alpha=0.3)
plt.show()

#calc residuals
residuals = y_test - y_test_pred

#plotting residuals

plt.figure(figsize=(12, 6)) #creates a new figure with a specific size (12x6 inches), allowing more space for visual clarity
plt.hist(residuals, bins=30, color='purple', alpha=0.7) #show how well the model's predictions align with actual values
plt.axvline(x=0, color='red', linestyle='--', linewidth=2) #drawing a line at 0
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.grid(alpha=0.3)
plt.show()
