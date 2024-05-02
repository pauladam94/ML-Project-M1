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

with open('our_data_x_kmeans.csv', newline='') as csvfile:
    x_data = np.loadtxt(csvfile, delimiter=',')
    x_data = x_data.reshape(-1, 28, 28)

with open('our_data_y_kmeans.csv', newline='') as csvfile:
    y_data = np.loadtxt(csvfile, delimiter=',')

x_data = [x.flatten() for x in x_data]

def detect(i, y):
    return str(int(i)) in str(int(y))

def create_svm(i, x_data, y_data, detect_function):
    print("Creating SVM model for digit ", i)
    y_detect = [detect_function(y) for y in y_data]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_detect, test_size=0.2)
    svm_model = svm.SVC()
    svm_model.fit(x_train, y_train)
    # We save the model

    with open('svm_'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    # We open the model
    with open('svm_'+str(i)+'.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    print("    Train accuracy =", svm_model.score(x_train, y_train))
    print("    Test accuracy  =", svm_model.score(x_test, y_test))

    prediction = svm_model.predict(x_data)

    number_errors = 0
    for i in range(len(x_data)):
        if prediction[i] != y_detect[i]:
            number_errors += 1
    print("    Number of errors =", number_errors)

for i in range(10):
    create_svm(str(i), x_data, y_data, lambda y: detect(i, y))

create_svm("two_numbers", x_data, y_data, lambda y: y >= 10)


"""
# We suppose that the image is a 28x28 numpy array
# And that the image is only 0 and 1.
def final_model(img):
    img = img.flatten()
    predictions = [0]*10
    for i in range(10):
        with open('svm_'+str(i)+'.pkl', 'rb') as f:
            svm_model = pickle.load(f)
        predictions[i] = svm_model.predict([img])
    with open('svm_two_numbers.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    predictions.append(svm_model.predict([img]))

    if predictions[10] == 0:
        for i in range(10):
            if predictions[i] == 1:
                return (predisctions, i)
    else:
        for i in reversed(range(10)):
            if predictions[i] == 1:
                return (predictions, 10 + i)
    return (predictions, 0)

number_errors = 0
for i in range(5000):
    (predictions, result) = final_model(x_data[i])
    real_value = y_data[i]
    # print("Prediction for image", i, ":", result, "Real value :", y_data[i])

    if result != real_value:
        number_errors += 1
        print("Error for image", i, ":", result, "Real value :", y_data[i])
        print(predictions)

print("Number of errors final prediction=", number_errors)

"""
