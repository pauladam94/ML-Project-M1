import numpy as np
import gzip

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import random as rd
from scipy import ndimage

from sklearn.cluster import KMeans

import csv
with open('our_data_x_kmeans.csv', newline='') as csvfile:
    x_train = np.loadtxt(csvfile, delimiter=',')
    #print(x_train)
    x_train = x_train.reshape(-1,28,28)

with open('our_data_y_kmeans.csv', newline='') as csvfile:
    y_train = np.loadtxt(csvfile, delimiter=',')
    #print(x_train)

def load_info_data_set():
    columns = 10
    rows = 1
    fig, ax = plt.subplots(rows, columns, squeeze=False)
    gs = fig.add_gridspec(1, columns, hspace=0, wspace=0)
    fig.suptitle('Several transformations of the same image')
    i = 0
    for r in range(rows):
        for c in range(columns):
            img_nb = rd.randint(0, 5001-1)
            img = x_train[img_nb]
            ax[0][c].imshow(img)
            ax[0][c].set_title(y_train[img_nb])
    for ax in fig.get_axes():
        ax.set_axis_off()
        ax.label_outer()
    fig.tight_layout()
    plt.show()



load_info_data_set()