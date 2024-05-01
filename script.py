import numpy as np
import gzip

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import random as rd
from scipy import ndimage

from sklearn.cluster import KMeans

import datetime

from tensorflow.keras.datasets import fashion_mnist, cifar10
from sklearn.model_selection import train_test_split

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import RandomCrop, RandomFlip, RandomRotation, RandomZoom, RandomRotation, RandomContrast, RandomBrightness, RandomTranslation
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

from tensorflow.keras.applications import ResNet50

import csv



with open('our_data_x.csv', newline='') as csvfile:
    x_train = np.loadtxt(csvfile, delimiter=',')
    #print(x_train)
    x_train = x_train.reshape(-1,28,28)


with open('our_data_y.csv', newline='') as csvfile:
    y_train = np.loadtxt(csvfile, delimiter=',')


data_shape = [28,28,1]


def build_CNN(layers=([32,64,64],[20]), input_shape=data_shape, output_dim=20, lr=0.9, data_augmentation=False, from_ResNet=False, trainable_CNN=True):

    model = Sequential()

    model.add(Input(shape=input_shape))

    # data augmentation
    if data_augmentation:
      pass

    # CNN layers
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=data_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=data_shape))


    # Flatten the data for dense layers

    model.add(Flatten())

    # Add dense layers
    model.add(Dense(40, activation='relu'))
    model.add(Dense(20, activation='relu'))


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model


cnn_model = build_CNN(lr=0.9)
cnn_model.summary()


# Split the training set into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.05, random_state=42)



# Display the dimensions of the split sets
print("Dimensions of the training set after splitting: ", len(x_train))
print("Dimensions of the validation set: ", len(x_valid))

tensorboard_callback = TensorBoard(log_dir='logs/small__' + datetime.datetime.now().strftime("%d-%m_%Hh%M"), histogram_freq=5)

history = cnn_model.fit(
        x_train, y_train,
        validation_data=(x_train, y_train),
        batch_size=100,
        epochs=30,
        callbacks=[tensorboard_callback])

cnn_model.evaluate(x_valid, y_valid, verbose=2)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

