from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#set threshold 
threshold = 0.55

#define a simple sequential model
def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation ='relu')
        Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#train the model
model.fit(X_train, y_train, epochs=10, batch_size = 32, validation_split=0.2)

predictions = model.predict(X_test)

binary_predictions = [1 if p>= threshold else 0 for p in predictions]
