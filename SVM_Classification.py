#SVM Classification

#Import the necessary libraries
import sys
from sys import argv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

#SVM Model
svm = SVC(kernel='rbf')
svm.fit(X_train,y_train)

#K-Fold Cross Validation 
scores = cross_val_score(svm, X_train, y_train, cv=5)

#Test the Model 
y_pred = svm.predict(X_test)
acc_score = accuracy_score(y_test,y_pred)
f1_score = f1_score(y_test, y_pred)

with open('Results.log','a') as f:
  f.write('\n' + "Classification SVM Model" + '\n')
  f.write('\n' + "The accuracy of the Classification SVM Model is: " + str(acc_score*100) +'\n')
  f.write("The f1 score of the Classification SVM Model is: " + str(f1_score*100) + '\n')
  f.close()

#Confusion Matrix for true positive and true negatives
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.savefig('SVM_Classification_Confusion_Matrix.jpg', format='jpg')
