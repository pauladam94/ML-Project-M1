import numpy as np
import gzip

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import random as rd
from scipy import ndimage

# Load the images from the compressed CSV file
with gzip.open('x_train.csv.gz', 'rb') as f:
    x_train = np.loadtxt(f, delimiter=',').astype(np.int64)
    x_train = x_train.reshape(-1, 32, 32, 3)
# Load the labels from the compressed CSV file
with gzip.open('y_train.csv.gz', 'rb') as f:
    y_train = np.loadtxt(f, delimiter=',', dtype=int)
    print(x_train)

def img2BW(img):
    x, y, c = img.shape
    img_ = np.zeros([x,y,1],dtype=np.int64)
    for i in range(x):
        for j in range(y):
            img_[i][j][0] = 0.2126*img[i][j][0]  + 0.7152*img[i][j][1] + 0.0722*img[i][j][2]
    return(img_)

def convol(img, ker):
    x, y, c = img.shape
    xk, yk = ker.shape
    img_ = np.zeros([x-xk//2,y-yk//2,1],dtype=np.int64)
    for i in range(x-xk//2):
        for j in range(y-yk//2):
            val = 0
            for ii in range(xk):
                for jj in range(yk):
                    val += img[i+ii-xk//2][j+jj-yk//2][0]*ker[ii][jj]
            img_[i][j][0] = val
    return(img_)

def threshold(img,v):
    x, y, c = img.shape
    img_ = np.zeros([x,y,1],dtype=np.int64)
    min=100000
    max=-100000
    for i in range(x):
        for j in range(y):
            if img[i][j][0]>max:
                max = img[i][j][0]
            if img[i][j][0]<min:
                min = img[i][j][0]
    f = max-min
    
    for i in range(x):
        for j in range(y):
            val = (img[i][j][0]-min)/f
            img_[i][j][0] = val*100
    #std =  ndimage.standard_deviation(img_)
    for i in range(x):
        for j in range(y):
            #val = np.sqrt(val)
            pass
    for i in range(x):
        for j in range(y):
            img_[i][j][0] = (img_[i][j][0])//10
    return(img_)

sobel = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
])

ker = np.array([
    [1,3,1],
    [3,10,3],
    [1,3,1]
])
ker = ker / np.sum(ker)

#ker = np.array([
#    [0,0,1,0,0],
#    [0,5,10,5,0],
#    [1,10,20,10,1],
#    [0,5,10,5,0],
#    [0,0,1,0,0]
#])
#ker = ker / np.sum(ker)




def load_info_data_set():
    # Load the images from the compressed CSV file
    with gzip.open('x_train.csv.gz', 'rb') as f:
        x_train = np.loadtxt(f, delimiter=',').astype(np.int64)
        x_train = x_train.reshape(-1, 32, 32, 3)
    # Load the labels from the compressed CSV file
    with gzip.open('y_train.csv.gz', 'rb') as f:
        y_train = np.loadtxt(f, delimiter=',', dtype=int)
    x_shape = x_train.shape
    y_shape = y_train.shape
    shape_image = x_train[0].shape

    print("Size X Train", x_shape)
    print("Size Picture :", x_train[0].shape)
    print("Size Y Train", y_shape)
    print("Y Train", y_train[0])
    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = 5
    for i in range(1, columns * rows + 1):
        img_nb = rd.randint(0, y_shape[0])
        fig.add_subplot(rows, columns, i)
        fig.axes[i-1].set_title(y_train[img_nb])
        fig.axes[i-1].set_axis_off()
        img = x_train[img_nb]
        img = img2BW(img)
        img = convol(img,ker)
        img = threshold(img, 0.1)
        plt.imshow(img, cmap='gray')
    plt.show()

load_info_data_set()