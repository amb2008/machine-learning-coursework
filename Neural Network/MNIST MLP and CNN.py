import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# call desired functions at end, 1 at a time

# get mnist dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

def plotSample():
  fig = plt.figure()
  fig.set_size_inches(10, 10)
  # get randomly selected images and plot
  for i in range(16):
      plt.subplot(4,4,i+1)
      num = np.random.randint(0, len(X_train))
      plt.imshow(X_train[num], cmap='gray')
      plt.title(f"Class {y_train[num]}")

  plt.tight_layout()

# make each pixel from 0 - 1 instead of 0 - 255
# only works for grayscale images
X_train = X_train / 255
X_test = X_test / 255

nb_classes = 10 # number of unique digits

# make y matrices with 0 in every 10 digits except the one that it actually is
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
print("Y_train shape", Y_train.shape)
print("Y_test shape", Y_test.shape)

# FULLY CONNECTED NEURAL NETWORK ---------

def MLP():
  # Two layers with 512 neurons and relu for linear modeling
  # Last layer with 10 output neurons and softmax for classifcation as a digit 0 - 9
  model = models.Sequential()
  # flatten to make into a vector
  # input shape is 28 x 28 because that is the size of the image in pixels
  model.add(layers.Flatten(input_shape=(28, 28)))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dense(10, activation='softmax'))
  # compile and train
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(X_test, Y_test))
  

# Convolutional NEURAL NETWORK ---------

def CNN(X_train, X_test):
  X_train = X_train.reshape(60000, 28, 28, 1) # add an additional dimension to represent the single-channel
  X_test = X_test.reshape(10000, 28, 28, 1)
  
  model = Sequential([
    # convolution neural network with 16 neurons and a 5x5 filter
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    # find the max val in each 2x2 cell
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='valid'),
    MaxPooling2D(pool_size=(2, 2)),
    #flatten into a vector so it can be passed to dense
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
  ])
  
  # compile and train
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X_train, Y_train, batch_size=128, epochs=7, validation_data=(X_test, Y_test))
  
  # get prediction
  y_hat = model.predict(X_test)
  
  # set num found, so we don't print every wrong answer
  # plot 16 incorrectly predicted numbers
  found = 0
  fig = plt.figure()
  fig.set_size_inches(14, 14)
  for i in range(10000):
    # check if predicted category is same in prediction and real
    if np.argmax(y_hat[i]) != np.argmax(Y_test[i]):
        plt.xticks([])
        plt.yticks([])
        found += 1
        plt.subplot(4,4,found)
        plt.imshow(X_test[i], cmap='gray')
        plt.title(f"Guess: {np.argmax(y_hat[i])}, Real: {np.argmax(Y_test[i])}")
    if found == 16:
      break

CNN(X_train, X_test)
plotSample()
MLP()
