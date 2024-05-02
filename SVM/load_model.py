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
from sklearn.model_selection import train_test_split
import csv
import pickle
from random import randint

with open('our_data_x_kmeans.csv', newline='') as csvfile:
    x_data = np.loadtxt(csvfile, delimiter=',')
    x_data = x_data.reshape(-1, 28, 28)

with open('our_data_y_kmeans.csv', newline='') as csvfile:
    y_data = np.loadtxt(csvfile, delimiter=',')

x_data = [x.flatten() for x in x_data]

# We suppose that the image is a 28x28 numpy array
# And that the image is only 0 and 1.
def final_model(img):
    img = img.flatten()
    predictions = [0]*10
    for i in range(10):
        with open('svm_'+str(i)+'.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        predictions[i] = svm_model.predict([img])[0]
    with open('svm_two_numbers.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    predictions.append(svm_model.predict([img])[0])
    if predictions[10] == 0:
        for i in range(10):
            if predictions[i] == 1:
                return (predictions, i)
    else:
        if predictions[0] == 1:
            return (predictions, 10)
        for i in reversed(range(10)):
            if predictions[i] == 1:
                return (predictions, 10 + i)
    return (predictions, randint(0, 9))

number_errors = 0
for i in range(len(x_data)):
    (predictions, result) = final_model(x_data[i])
    real_value = y_data[i]
    # print("Prediction for image", i, ":", result, "Real value :", y_data[i])

    if result != real_value:
        number_errors += 1
        # if predictions != [False, True, False, False, False, False, False, False, False, False, True]:
        print("Error for image", i, ", Prediction :", result, "Real value :", y_data[i])
        # iprint(predictions)

print("Number of errors final prediction=", number_errors)