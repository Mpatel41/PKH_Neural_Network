#SVM Classification

import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Read in the Data
dataset = pd.read_excel("out.xlsx")

#Manipulate the Dataset
dataset['Labels'] = dataset['totflux_A_1000']
dataset.loc[dataset['Labels'] <= 0, 'Labels' ] = 0
dataset.loc[dataset['Labels'] > 0, 'Labels' ] = 1
classification_dataset = dataset.drop(columns=['totflux_A_1000','Deff_membrane','Unnamed: 0','job_id'])
regression_dataset = dataset.drop(columns=['Labels','Deff_membrane','Unnamed: 0','job_id'])
classification_dataset.info()

#Define the input features and the labels for Classification Dataset
X_class = classification_dataset.iloc[:,:5].to_numpy() #Input Features
y_class = classification_dataset.iloc[:,5].to_numpy() #Output

#Split into Training and Testing Steps
X_train, X_test, y_train, y_test = train_test_split(X_class,y_class, random_state=2, test_size=0.20, shuffle=True)
print(y_train)

#Standarize the mean=0 and variance=1
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#SVM Classification Model
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

#SVM Model
svm = SVC(kernel='rbf')
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
acc_score = accuracy_score(y_test,y_pred)
f1_score = f1_score(y_test, y_pred)
print("The accuracy score of SVM Model is: " + acc_score) 
print("The F1 score of the SVM Model is: " + f1_score) 


#Confusion Matrix for true positive and true negatives
cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
