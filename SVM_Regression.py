#SVR model
#support vector regression

#Import the necessary libraries 
import sys
from sys import argv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

#Get either the actual data or test data based on what was provided 
if len(sys.argv) != 2:
    print("Usage: python file.py <folder1>")
    sys.exit(1)

#Get the folder paths from above
data = sys.argv[1]

#Read in the File 
dataset = pd.read_csv(data)

#Define the input features and the labels for Classification Dataset
X_class = dataset.iloc[:,:5].to_numpy() #Input Features
y_class = dataset.iloc[:,5].to_numpy() #Output

#Split into Training and Testing Steps
X_train, X_test, y_train, y_test = train_test_split(X_class,y_class, random_state=2, test_size=0.20, shuffle=True)

#Standarize the mean=0 and variance=1
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Support Vector Regression Model 
svr = SVR(kernel='rbf', gamma=0.1, C=1.0)
svr.fit(X_train, y_train)

#K-Fold Cross Validation 
scores = cross_val_score(svr, X_train, y_train, cv=5)

#Test SVR on testing set 
predictions = svr.predict(X_test)

#MSE Score 
y_pred = svr.predict(X_test)  #predictions using SVR

# calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

#Write MSE to log file
with open('Results.log','a') as f:
  f.write('\n' + "Support Vector Regression Model")
  f.write('\n' + "The Mean-squared error of the Regression SVM Model is: " + str(mse) +'\n')
  f.close()
