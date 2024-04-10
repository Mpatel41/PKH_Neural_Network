#Regression Neural Network 

#Import Necessary Packages 
import pandas as pd
import tensorflow
import keras
import scikeras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Define base model
def baseline_model():
  model = Sequential()
  model.add(Dense(5, input_shape=(X_train.shape[1],), kernel_initializer='normal', activation='relu')) #5 input features = 5 neurons starting
  model.add(Dense(20, activation='softmax', kernel_initializer='normal')) #20 neurons 
  model.add(Dense(1)) #1 output neuron
 # Compile model
  model.compile(loss='mse', optimizer='adam', metrics='mse') #mean-squared error loss and metric 
  return model

model = baseline_model() 
model.summary() #Returns the number of neurons and the format shape 

#Fit the Model on training data 
EPOCHs = 20 
batch_size = 32

model = baseline_model()
fitted_model = model.fit(X_train,y_train,batch_size=batch_size,epochs=EPOCHs,verbose=0,shuffle=True)


#K-fold Cross Validation (does not work properly) 
estimator = KerasRegressor(model=baseline_model, epochs=10, batch_size=10, verbose=0)
kfold = KFold(n_splits=3)
results = cross_val_score(estimator, X, y, cv=kfold, scoring='neg_mean_squared_error')
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

#Test the Model on test data 
test = model.predict(X_test)
print(test)

