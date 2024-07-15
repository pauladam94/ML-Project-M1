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
    x_data = np.loadtxt(csvfile, delimiter=',')
    x_data = x_data.reshape(-1, 28, 28)

with open('our_data_y.csv', newline='') as csvfile:
    y_data = np.loadtxt(csvfile, delimiter=',')

x_data = [x.flatten() for x in x_data]
    


def detect(n,d):
    return str(d) in str(n)

def create_svm(i, x_data, y_data):
    y_detect = [detect(y,i) for y in y_data]
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_detect, test_size=0.2)
    svm_model = svm.SVC()
    svm_model.fit(x_train, y_train)
    # We save the model
    import pickle
    with open('svm_'+str(i)+'.pkl','wb') as f:
        pickle.dump(svm_model,f)
    # We open the model
    with open('svm_'+str(i)+'.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    print("\n\n\n")
    print("Train accuracy =", svm_model.score(x_train, y_train))
    print("Test accuracy  =", svm_model.score(x_test, y_test))

    prediction = svm_model.predict(x_data)

    for i in range(4995):
        if prediction[i] != y_data[i]:
            print(i)

for i in range(10):
    create_svm(i, x_data, y_data)