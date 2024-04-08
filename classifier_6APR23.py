import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#load the dataset
df = pd.read_csv('out.csv')

#last column is the output and 2-6 are features
X = df.iloc[:, 2:-1].values #skip first two columns
y = df.iloc[:, -1].values

#convert continuous outputs to binary class
y = np.where(y > 0.55, 1, 0) #threshold = 0.55 for now... adjust accordingly

#data already split into X_train, X_test, y_train, and y_test (80% training 20% test)
#already normalized the features with StandardScaler()

#define a model
def create_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy') #changed loss from mean squared error since we just have 0s and 1s, but idk if this is right
    return model

#wrap the model with KerasClassifier
estimator = KerasClassifier(model=create_model, input_shape=X_train.shape[1], epochs=100, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold, scoring='binary_crossentropy')
print("Baseline: %.2f (%.2f) MSE)" % (results.mean(), results.std()))


