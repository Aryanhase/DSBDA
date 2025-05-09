import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading dataset
df = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\DSBDA\logisticregression5\Social_Network_Ads.csv")
df

#preprocessing
df['Gender'] #displays gender col
df.isnull().sum() #checks for missing vals in all cols
df.dtypes #dtypes of all cols

df['Gender'] = df['Gender'].map({'Male':1, 'Female':0}) #converts categorical values (Male, Female) to numerical values (1, 0)
df['Gender']
df

#train test split; x-features and y-target
X = df.drop(['Purchased', 'User ID'], axis=1) #drops Purchased and User ID columns as they are not relevant for prediction
y = df['Purchased'] 

X
y

#splitting the datasety into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0) #25% of the data is used for testing, 75% for training

#scaling the features
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train) #fits the scaler to the training data and transforms it
X_test = ss.transform(X_test) #applies the same transformation to the test data
X_test
X_train
y_train

#using naive bayes theorem
from sklearn.naive_bayes import GaussianNB #model assumes that the features follow a Gaussian (normal) distribution
gaussian = GaussianNB() 
gaussian.fit(X_train, y_train) #trains the model on the training set

#making predictions
y_pred = gaussian.predict(X_test) #makes predictions on the test set using the trained Naive Bayes model

#model accuracy
gaussian.score(X_test,y_test) #calculates the accuracy of the model on the test set

from sklearn.metrics import precision_score,confusion_matrix,accuracy_score,recall_score

accuracy = accuracy_score(y_test,y_pred) #measures the proportion of correct predictions
print(accuracy)

precision =precision_score(y_test, y_pred,average='micro') #measures the proportion of positive predictions that were actually positive avg micro computes the global precision across all classes
print(precision)

recall = recall_score(y_test, y_pred,average='micro') #measures the proportion of actual positives that were correctly identified
print(recall)


cm = confusion_matrix(y_test, y_pred)
cm

#plotting the Confusion Matrix as a Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
