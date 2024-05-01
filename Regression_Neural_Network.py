#Regression Neural Network

#Import Necessary Packages
import sys
from sys import argv
import tensorflow
import keras
import scikeras
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import regularizers

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

# Define base model
def baseline_model():
  model = Sequential()
  model.add(Dense(5, input_shape=(X_train.shape[1],), kernel_initializer='normal',kernel_regularizer=regularizers.l2(0.01),activation='linear')) #5 input features = 5 neurons starting
  model.add(Dense(100, activation='linear',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='normal'))
  model.add(Dense(50, activation='linear',kernel_regularizer=regularizers.l2(0.01),kernel_initializer='normal')) #20 neurons
  model.add(Dense(1)) #1 output neuron
  model.compile(loss='mean_squared_error', optimizer='adam') #mean-squared error loss and metric
  return model

#Assign the model to a variable 
model = baseline_model()

#K-fold Cross Validation 
regressor = KerasRegressor(model=baseline_model, epochs=40, batch_size=32, verbose=0)
kfold = KFold(n_splits=5)
results = cross_val_score(regressor, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

#Fit the model on training data set 
fitted_model = model.fit(X_train,y_train,batch_size=32,epochs=40,verbose=0,shuffle=True)

#Test the Model on test data
test = model.predict(X_test)

#Get the Mean Squared Error (MSE) scores 
MSE_scaled = mean_squared_error(y_test,test)

#Write the MSE Scores to log file 

with open('Results.log','a') as f:
  f.write('\n' + "Regression Neural Network Model" + '\n')
  f.write("The Mean-squared error of the Regression Neural Network Model is: " + str(MSE_scaled) +'\n')
  f.close()


