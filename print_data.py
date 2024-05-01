import numpy as np
import gzip

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import random as rd
from scipy import ndimage

from sklearn.cluster import KMeans

import csv
with open('our_data_x.csv', newline='') as csvfile:
    x_train = np.loadtxt(csvfile, delimiter=',')
    #print(x_train)
    x_train = x_train.reshape(-1,26,26)

with open('our_data_y.csv', newline='') as csvfile:
    y_train = np.loadtxt(csvfile, delimiter=',')
    #print(x_train)

def load_info_data_set():

    columns = 20
    fig = plt.figure()
    gs = fig.add_gridspec(1, columns, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    fig.suptitle('Several transformations of the same image')

    for i in range(0, columns):
        r = len(y_train)
        img_nb = rd.randint(0, r-1)
        img = x_train[img_nb]
        axs[0][i].imshow(img)
        axs[0][i].set_title(y_train[img_nb])
        j = 0

    for ax in fig.get_axes():
        ax.set_axis_off()
        ax.label_outer()
    fig.tight_layout()
    plt.show()

load_info_data_set()