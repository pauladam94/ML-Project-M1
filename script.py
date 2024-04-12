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



with open('our_data.csv', newline='') as csvfile:
    x_train = np.loadtxt(csvfile, delimiter=',').astype(np.int8)
    x_train = x_train.reshape(-1,26,26)

# print(x_train[0])

# pltfig = plt.figure()
# plt.imshow(x_train[0])
# plt.show()

data_shape = [26,26,1]


def build_CNN(layers=([32,64,64],[64]), input_shape=data_shape, output_dim=20, lr=0.01, data_augmentation=False, from_ResNet=False, trainable_CNN=True):
    """
    Function to construct a Convolutional Neural Network (CNN).

    Args:
    layers: Tuple of lists specifying the number of filters for each convolutional layer and
            the number of units for each dense layer. Default is ([32,64,64],[64]).
    input_shape: Tuple representing the dimensions of the input data (height, width, channels).
    output_dim: Integer indicating the number of output classes.
    lr: Float representing the learning rate for the Adam optimizer. Default is 0.001.
    data_augmentation: Boolean flag indicating whether data augmentation should be applied.
                      Default is False.
    from_ResNet: Boolean flag indicating whether to use pre-trained ResNet50 layers.
                 Default is False.
    trainable_CNN: Boolean flag indicating whether CNN layers should be trainable.
                   Default is True.

    Returns:
    model: A compiled Keras model.
    """
    model = Sequential()

    model.add(Input(shape=input_shape))

    # data augmentation
    if data_augmentation:
      pass

    # CNN layers
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=data_shape))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2,2))




    # Flatten the data for dense layers

    model.add(Flatten())

    # Add dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(20, activation='relu'))
    
    # Output layer


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=lr), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model


cnn_model = build_CNN(lr=0.01)
cnn_model.summary()




with gzip.open('y_train.csv.gz', 'rb') as f:
    y_train = np.loadtxt(f, delimiter=',', dtype=int)
    #print(x_train)


# Split the training set into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=42)



# Display the dimensions of the split sets
print("Dimensions of the training set after splitting: ", len(x_train))
print("Dimensions of the validation set: ", len(x_valid))

tensorboard_callback = TensorBoard(log_dir='logs/small__' + datetime.datetime.now().strftime("%d-%m_%Hh%M"), histogram_freq=5)

history = cnn_model.fit(
        x_train, y_train,
        validation_data=(x_train, y_train),
        batch_size=90,
        epochs=25,
        callbacks=[tensorboard_callback])

cnn_model.evaluate(x_valid, y_valid, verbose=2)
plot_graphs(history)


