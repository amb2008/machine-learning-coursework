import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint

# create a dataframe from the csvs
feature = pd.read_csv('https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_label.csv')
X = feature.values
y = label.values

# normalize the data using sklearn's StandardScaler
# for normal distribution

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# split the scaled data in validation and train
X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.1, random_state=3)

# input shape = (num features, )
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    # output of one neuron with linear activation
    Dense(1, activation='linear')
])

# find best val_loss epoch
check_filepath = '/tmp/checkpoint'
checkpoint = ModelCheckpoint(
    filepath = check_filepath,
    save_best_only = True,
    monitor = 'val_loss',
    mode = 'min',
    save_weights_only=True,

)

model.compile(    
    optimizer='adam', # 'rmsprop', 'sgd'
    loss='mean_squared_error', # 'mae'
    metrics='mse'
) # use the Adam optimizer

# print a summary of the model
model.summary()

# train the model (use the train data and validation data from above)
# callbacks=checkpoint saves the best model according to our checkpoint definition
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2000, batch_size=64, callbacks=checkpoint)

# load weights from saved best model
model.load_weights(check_filepath)

def plotLoss():
  # plot the train and validation losses on the same picture
  # make sure to label the axis and create a legend
  train_loss_history = history.history['loss']
  val_loss_history = history.history['val_loss']
  plt.figure(figsize=(8,6))
  # semilog creates the 10^1 sides
  plt.semilogy(train_loss_history, label="train")
  plt.semilogy(val_loss_history, label="val")
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.grid()
  plt.legend()

# load testing data
X_test = pd.read_csv('https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_test_feature.csv').values
y_test = pd.read_csv('https://raw.githubusercontent.com/ajn313/NYU2023SummerML3/main/Day5/fish_market_test_label.csv').values

# scale the test data using the scaler above
Xtest_s = scaler.transform(X_test)

# predict the corresponding y_hat value of the test dataset (use the scaled test data)
y_hat = model.predict(Xtest_s)

plt.figure(figsize=(8,6))
plt.scatter(np.arange(y_hat.shape[0]), y_hat, label='Prediction')
plt.scatter(np.arange(y_test.shape[0]), y_test, label='Ground Truth')
plt.legend()
plt.xlabel('data no.')
plt.ylabel('predicted value')
plt.grid()

# print MSE, RMSE (root-mse), MAE on the train and test data
# compare these results against last week's results (when we used linear/polynimial regression)
MSE1 = train_loss_history[-1]
RMSE1 = np.sqrt(MSE1)
MSE2 = np.mean((y_test-y_hat)**2)
RMSE2 = np.sqrt(MSE2)
print(MSE1, RMSE1)
print(MSE2, RMSE2)








