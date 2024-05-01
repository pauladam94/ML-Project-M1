from sklearn import svm
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

x_train = [x.flatten() for x in x_train]
y_train = [1 if (y == 1 or y<=10) else 0 for y in y_train]


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

isThereOne = svm.SVC()
isThereOne.fit(x_train, y_train)

print("\n\n\n")

print("Train accuracy =",isThereOne.score(x_train, y_train))
print("Test accuracy =",isThereOne.score(x_test, y_test))


import matplotlib.pyplot as plt
import numpy as np

# Predict the labels
y_pred = isThereOne.predict(x_test)

# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 

# Select 10 random images and their true and predicted labels
indices = np.random.choice(range(x_test.shape[0]), size=10, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')

plt.tight_layout()
plt.show()