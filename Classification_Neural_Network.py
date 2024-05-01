#Import the necessary libraries 
import sys
from sys import argv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

#Defining a Neural Network Classification Model 
def create_model(input_shape):
    model = Sequential([
        Dense(5, activation='relu', input_shape=(input_shape,)),
        Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='normal'),
        Dense(100, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='normal'),
        Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy']) 
    return model

#K-Fold Cross Validation on the model with KerasClassifier on Training Set 
estimator = KerasClassifier(model=create_model, input_shape=X_train.shape[1], epochs=40, verbose=0)
kfold = KFold(n_splits=5)
results = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring='accuracy')

#Assign the model to a variable
model = create_model(5)

#Fit the Model on Training Data
fitted_model = model.fit(X_train,y_train,batch_size=5,epochs=20,verbose=0,shuffle=True)

#Test the Model on testing set 
test = model.predict(X_test)

#Convert the probabilites predicted to binary outcomes that match the labels 
threshold = 0
binary_predictions = np.where(test >= threshold, 1, 0)

#Calculating the Accuracy and F1 score of the predictions 
acc = accuracy_score(y_test, binary_predictions)
score = f1_score(y_test, binary_predictions)

with open('Results.log','a') as f:
  f.write('\n' + "Classification Neural Network Model" + '\n')
  f.write("The accuracy of the Classification Neural Network Model is: " + str(acc*100) +'\n')
  f.write("The f1 score of the Classification Neural Network Model is: " + str(score*100) + '\n')
  f.close()
