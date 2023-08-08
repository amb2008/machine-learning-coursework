import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

model = Sequential([
    # 16 neurons in first and second layers, connected by relu
    # if you increase num of neurons in layers, increase by powers of 2
    # first layer has an input with one column and no rows
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(16, activation='relu'),
    # output of one neuron with linear activation
    Dense(1, activation='linear')
])

# implement adam optimizer based on MSE loss function, and report MSE
model.compile(
    optimizer='adam', # 'rmsprop', 'sgd'
    loss='mean_squared_error', # 'mae'
    metrics='mse'
)

# create cosine + sine wave with noise for training and testing
X_train = tf.random.uniform(shape=[1000, ], minval=0, maxval=12)
Y_train = X_train * tf.cos(X_train) + tf.sin(X_train) ** 2 + 0.5*tf.random.normal(shape=[1000, ])

# create a similar cosine + sine wave with different noise for training and testing
X_test = tf.random.uniform(shape=[500, ], minval=0, maxval=12)
Y_test = X_test * tf.cos(X_test) + tf.sin(X_test) ** 2 + 0.5*tf.random.normal(shape=[500, ])

X_train = tf.reshape(X_train, (-1, 1))
Y_train = tf.reshape(Y_train, (-1, 1))
X_train.shape, Y_train.shape

# training_history = model.fit(features, labels, epochs=epochs, batch_size=batch_size)
# batch size is amount of data points taken at a time
history = model.fit(X_train, Y_train, epochs=1000, validation_split=0.1, batch_size=32)

# prediction = model.predict(features)
y_hat = model.predict(X_test)

# plot test in comparison to test
plt.figure(figsize=(8,6))
plt.scatter(X_test, Y_test, label='Ground Truth')
plt.scatter(X_test, y_hat, marker='x', label='Prediction')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

# history.history is calling 
loss_history = history.history['loss']

def plotLoss():
  # plot loss history
  plt.figure(figsize=(8,6))
  # semilog creates the 10^1 sides
  plt.semilogy(loss_history)
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.grid()

