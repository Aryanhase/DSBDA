import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
#loading dataset
df = pd.read_csv("Social_Network_Ads.csv")
df

#preprocessing
df['Gender']
df.isnull().sum() #null values in each col and sum is returned 
df.dtypes #dtypes of cols

df['Gender'] = df['Gender'].map({'Male':1, 'Female':0}) #converts categorical values (Male, Female) to numerical values (1, 0).
#ml model only takes numerical values
df['Gender']
df

# train test split; x-features and y-target
X = df.drop(['Purchased', 'User ID'], axis=1) #drops Purchased and User ID as they are not features.
y = df['Purchased'] #purchased col is target variable

X
y

#splitting the dataset into train and test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=0) #25% of the data is used for testing, 75% for training. random_state=0 ensures reproducibility of the split

#scaling the features; ensures that all features contribute equally to the model

from sklearn.preprocessing import StandardScaler
#age and salary can have vastly diff scales
ss = StandardScaler()
X_train = ss.fit_transform(X_train) #fits the scaler to X_train and applies the transformation
X_test = ss.transform(X_test) #applies the same transformation to X_test.

X_train

#model training

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0) #classification algorithm used for binary outcomes random_state ensures same initializzation for reproducibility
classifier.fit(X_train, y_train)

#predicting test set results
y_pred = classifier.predict(X_test) #predicts the target variable (Purchased) for the test set using the trained model.

#model accuracy score
classifier.score(X_test, y_test)

#evaluating the performance using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#plotting the Confusion Matrix
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
