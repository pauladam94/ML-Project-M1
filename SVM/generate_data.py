import numpy as np
import gzip

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import random as rd
from scipy import ndimage

from sklearn.cluster import KMeans

import csv


# Load the images from the compressed CSV file
with gzip.open('x_train.csv.gz', 'rb') as f:
    x_train = np.loadtxt(f, delimiter=',').astype(np.int64)
    x_train = x_train.reshape(-1, 32, 32, 3)
# Load the labels from the compressed CSV file
with gzip.open('y_train.csv.gz', 'rb') as f:
    y_train = np.loadtxt(f, delimiter=',', dtype=int)
    #print(y_train)

def img2BW(img):
    x, y, c = img.shape
    img_ = np.zeros([x,y,1],dtype=np.int64)
    for i in range(x):
        for j in range(y):
            img_[i][j][0] = 0.333*img[i][j][0]  + 0.333*img[i][j][1] + 0.333*img[i][j][2]
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


# -------- DATA AUGMENTATION FUNCTIONS --------

def shift_up(img):
    x, y = img.shape
    img_ = np.zeros([x,y],dtype=np.int64)
    img_[x-1] = img[0]
    for i in range(x-1):
        img_[i] = img[i+1]
    return(img_)


def shift_down(img):
    x, y = img.shape
    img_ = np.zeros([x,y],dtype=np.int64)
    img_[0] = img[x-1]
    for i in range(x-1):
        img_[i+1] = img[i]
    return(img_)

def shift_left(img):
    x, y = img.shape
    img_ = np.zeros([x,y],dtype=np.int64)
    img_[x-1] = img[0]
    for i in range(x):
        img_[i][y-1] = img[i][0]
        for j in range(y-1):
            img_[i][j] = img[i][j+1]
    return(img_)

def shift_right(img):
    x, y = img.shape
    img_ = np.zeros([x,y],dtype=np.int64)
    img_[x-1] = img[0]
    for i in range(x):
        img_[i][0] = img[i][y-1]
        for j in range(y-1):
            img_[i][j+1] = img[i][j]
    return(img_)


def kmeans(image, n_clusters):
    # Reshape the image to be a list of pixels
    image_3d = np.zeros((image.shape[0] * image.shape[1], 3))

    # first column: x pixel value
    # second column: y pixel value
    # third column: luminance pixel value
    for i in range(image.shape[0] * image.shape[1]):
        image_3d[i, 0] = i % image.shape[0]
        image_3d[i, 1] = i // image.shape[0]
        image_3d[i, 2] = image[i % image.shape[0], i // image.shape[0]][0]

    # Perform k-means clustering
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(image_3d)

    # Get the label of each cluster
    labels = k_means.labels_
    centers = k_means.cluster_centers_
    labels_center = np.zeros(n_clusters)
    for i in range(n_clusters):
        labels_center[i] = image_3d[labels == i][:, 2].mean()

    segmented_image = np.zeros(image.shape)
    for i in range(image.shape[0] * image.shape[1]):
        x = i % image.shape[0]
        y = i // image.shape[0]
        segmented_image[x, y] = labels_center[labels[i]]
    return segmented_image



def kmeans2(image):
    n_clusters = 2
    # Reshape the image to be a list of pixels
    image_3d = np.zeros((image.shape[0] * image.shape[1], 3))

    # first column: x pixel value
    # second column: y pixel value
    # third column: luminance pixel value
    for i in range(image.shape[0] * image.shape[1]):
        image_3d[i, 0] = i % image.shape[0]
        image_3d[i, 1] = i // image.shape[0]
        image_3d[i, 2] = image[i % image.shape[0], i // image.shape[0]][0]

    # Perform k-means clustering
    k_means = KMeans(n_clusters=2)
    k_means.fit(image_3d)

    # Get the label of each cluster
    labels = k_means.labels_
    centers = k_means.cluster_centers_
    labels_center = np.zeros(n_clusters)
    for i in range(n_clusters):
        labels_center[i] = image_3d[labels == i][:, 2].mean()
    max_mean = -100000
    min_mean =  100000
    for i in range(n_clusters):
        max_mean, min_mean = max(max_mean, labels_center[i]), min(min_mean, labels_center[i])
    mean_mean = (max_mean + min_mean)/2
    
        

    segmented_image = np.zeros(image.shape)
    for i in range(image.shape[0] * image.shape[1]):
        x = i % image.shape[0]
        y = i // image.shape[0]
        segmented_image[x, y] = 0 if labels_center[labels[i]]<mean_mean else 1
    return segmented_image

ker3 = np.array([
    [0,3,0],
    [3,10,3],
    [0,3,0]
])
ker3 = ker3 / np.sum(ker3)

# ker5 = np.array([
#     [0,0, 1, 2, 1],
#     [0,0, 10,5, 0],
#     [1,7,20,7,1],
#     [0,5 ,10,0, 0],
#     [1,2, 1, 0, 0]
# ])
# ker5 = ker5 / np.sum(ker5)

ker5 = np.array([
    [0,0, 0, 1, 1],
    [0,0, 2,2, 0],
    [0,2, 15,2,0],
    [0,2 ,2,0, 0],
    [1,1, 0, 0, 0]
])
ker5 = ker5 / np.sum(ker5)



def load_info_data_set(transformation):
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

    #print("Size X Train", x_shape)
    #print("Size Picture :", x_train[0].shape)
    #print("Size Y Train", y_shape)
    #print("Y Train", y_train[0])

    columns = 20
    rows = len(transformation)
    fig = plt.figure()
    gs = fig.add_gridspec(rows + 1, columns, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    fig.suptitle('Several transformations of the same image')

    for i in range(0, columns):
        img_nb = rd.randint(0, y_shape[0]-1)
        img = x_train[img_nb]
        axs[0][i].imshow(img)
        axs[0][i].set_title(y_train[img_nb])
        j = 0
        for transf in transformation:
            j += 1
            #print("Transformation ", j)
            axs[j][i].imshow(transf(img), cmap='gray')

    for ax in fig.get_axes():
        ax.set_axis_off()
        ax.label_outer()
    fig.tight_layout()
    plt.show()


def denoise(img): # Taille image 32*32*3 -> 28*28*1
    img_ = kmeans2(convol(convol(convol(img2BW(img),ker3),ker5),ker3))
    #img_ = convol(convol(convol(img2BW(img),ker3),ker5),ker3)
    #print(np.squeeze(img_).shape)
    return (np.squeeze(img_))
    
#load_info_data_set([denoise])
    

#load_info_data_set([
#    lambda img : kmeans(convol(convol(convol(convol(img2BW(img),ker3),ker5),ker5),ker3), 2),
#    lambda img : convol(convol(convol(convol(img2BW(img),ker3),ker5),ker5),ker3)
#])



with open('our_data_y.csv', 'w') as f:
    i = 0
    for n in y_train:
        for _ in range(5):
            f.write(str(n) + '\n')
        print(i)
        i += 5

    

with open('our_data_x.csv', 'w') as f:
    i = 0
    for img in x_train:
        img_ = denoise(img)
        f.write(','.join(map(str, img_.flatten().tolist())) + '\n')
        f.write(','.join(map(str, shift_down(img_).flatten().tolist())) + '\n')
        f.write(','.join(map(str, shift_up(img_).flatten().tolist())) + '\n')
        f.write(','.join(map(str, shift_left(img_).flatten().tolist())) + '\n')
        f.write(','.join(map(str, shift_right(img_).flatten().tolist())) + '\n')
        print(i)
        i += 5