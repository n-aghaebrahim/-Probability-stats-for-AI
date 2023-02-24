import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# load the data into a pandas DataFrame
df = pd.read_csv('winequality-red.csv',sep=';')
#df = pd.read_csv('winequality.csv')

# split the data into features and target variables
X = df.drop('quality', axis=1)
y = df['quality']

# create binary class labels from the target variable
y = np.where(y >= 6, 1, 0)

# standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build the model using Keras
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# evaluate the model's accuracy on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy:', test_accuracy)
