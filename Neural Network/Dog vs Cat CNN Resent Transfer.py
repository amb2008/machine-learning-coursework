import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras import layers, models, applications, optimizers, losses
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# PREPARE DATA


# download cats vs dogs dataset
setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")
(train_ds, test_ds), metadata = tfds.load('cats_vs_dogs', split=['train[:35%]', 'train[90%:95%]'],
                                          with_info=True, as_supervised=True, shuffle_files=True,)

# images are different size, so we must preprocess
def image_formatting(image, label):
    # converting pixel values (uint8) to float32 type
    image = tf.cast(image, tf.float32)
    # normalizing the data to be in range of -1, +1
    image = applications.resnet_v2.preprocess_input(image)
    # resizing all images to a shape of 224x*224*3
    image = tf.image.resize(image, (224, 224))
    return image, label

# converting all images to same shape and formatting them for quicker training of the model
train = train_ds.map(image_formatting)
test = test_ds.map(image_formatting)

# suffle the data and create batches
batch_size = 128
shuffle_size = 1000
train_batches = train.shuffle(shuffle_size).batch(batch_size)
test_batches = test.batch(batch_size)


# CREATE AND TRAIN MODEL


# create the base model based on ResNet152V2
base_model = applications.ResNet152V2(
    weights='imagenet',  # Load weights pre-trained on the ImageNet dataset
    input_shape=(224, 224, 3),
    include_top=False) # do not include the classifier at the top

model = models.Sequential()
# add the entire base_model as "first layer"
model.add(base_model)
# add a GlobalAveragePooling2D layer, which average pools and flattens data
model.add(layers.GlobalAveragePooling2D())
# add to the model a Dense layer with 256 neurons and ReLu activation
model.add(layers.Dense(256, activation='relu'))
# add to the model a Dense layer with 1 neurons and Sigmoid activation
model.add(layers.Dense(1, activation='sigmoid'))
# do not train the first layer (ResNet/base_model) of the model as it is already trained
model.layers[0].trainable = False

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model for 5 epochs: use a batch size of 128 data samples
# set epoch to 5 for better results, but it will take longer
history = model.fit(train_batches, batch_size=128, epochs=1, workers=32, validation_data=(test_batches))


# GET PREDICTIONS AND SHOW ALL INCORRECTLY LABELED PICTURES


# get prediction
# seperate test x and y
y = np.concatenate([y for x, y in test_batches], axis=0)
ximage = np.concatenate([x for x, y in test_batches], axis=0)

# make prediction
y_hat = model.predict(ximage)

y = y.reshape(-1, 1)

plt.figure(figsize=(12,12))
found = 0

for i in range(y.shape[0]):
  plt.xticks([])
  plt.yticks([])
  if y[i] < 0.5 and y_hat[i] > 0.5:
    found+=1
    plt.subplot(4,4,found)
    plt.imshow(ximage[i])
    if y[i] == [1]:
      label = "dog"
    else:
      label = "cat"
    plt.title(label)
  if y[i] > 0.5 and y_hat[i] < 0.5:
    found+=1
    plt.subplot(4,4,found)
    plt.imshow(ximage[i])
    if y[i] == [1]:
      label = "dog"
    else:
      label = "cat"
    plt.title(label)

print("\n\n WRONG GUESSES (correctly labeled on image):")
