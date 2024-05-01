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


def detect(d, n):
    return str(d) in str(n)

x_data = [x.flatten() for x in x_data]

y_data = [detect(y,1) for y in y_data]



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

isThereOne = svm.SVC()
isThereOne.fit(x_train, y_train)





# We save the model
import pickle
with open('isThereOne.pkl','wb') as f:
    pickle.dump(isThereOne,f)


# We open the model
with open('isThereOne.pkl', 'rb') as f:
    isThereOne = pickle.load(f)



print("\n\n\n")

print("Train accuracy =", isThereOne.score(x_train, y_train))
print("Test accuracy  =", isThereOne.score(x_test, y_test))



prediction = isThereOne.predict(x_data)

for i in range(4995):
    if prediction[i] != y_data[i]:
        print(i)

# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# isThere = [svm.SVC() for _ in range(1)]

# for i in range(1):
#     print(y_train)
#     print(i)
#     isThere[i].fit(x_train, [detect(i,y) for y in y_train])





# # We save the model (test)
# import pickle
# with open('isThere.pkl','wb') as f:
#     pickle.dump(isThere[0],f)


# # We open the model
# with open('isThere.pkl', 'rb') as f:
#     isThere[0] = pickle.load(f)



# print("\n\n\n")

# for i in range(1):
#     print(i)
#     print("Train accuracy =", isThere[i].score(x_train, y_train))
#     print("Test accuracy  =", isThere[i].score(x_test, y_test))
#     print("\n")



# prediction = isThere[i].predict(x_data)

# for i in range(4995):
#     if prediction[i] != y_data[i]:
#         print(i)