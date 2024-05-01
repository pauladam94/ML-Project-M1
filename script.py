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
    x_train = x_train.reshape(-1, 28, 28)

with open('our_data_y.csv', newline='') as csvfile:
    y_train = np.loadtxt(csvfile, delimiter=',')

# We build a CNN model
# We train on hand writing digits from 0 to 19

data_shape = [28, 28, 1]

# We split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42)

# We normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# We add a channel dimension

x_train = np.expand_dims(x_train, axis=-1)

x_test = np.expand_dims(x_test, axis=-1)

# We build the model

model = Sequential()
model.add(Input(shape=data_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# We train the model

model.fit(x_train, y_train, epochs=30, batch_size=64,
          validation_data=(x_test, y_test))

# We evaluate the model

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# We save the model
#model.save('cnn_model.h5')