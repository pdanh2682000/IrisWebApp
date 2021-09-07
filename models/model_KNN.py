# Importing necessary libraries
import pandas as pd # To manage data as data frames
import numpy as np # To manipulate data as arrays
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Importing the dataset
data = pd.read_csv("./iris.csv")
# print(data)

# Dictionary containing the mapping
variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Encoding the target variables to integers
data = data.replace(['Setosa', 'Versicolor' , 'Virginica'], [0, 1, 2])

X = data.iloc[:, 0:-1] # Extracting the features/independent variables
# print(X)
y = data.iloc[:, -1] # Extracting the target/dependent variable
# print(y)

logreg = KNeighborsClassifier(n_neighbors=3)
logreg.fit(X, y) # tap hoc du lieu

def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # 
    prediction = variety_mappings[logreg.predict(query)[0]] # Retrieve from dictionary
    return prediction # Return the prediction

# print(classify(0,1,0,1));