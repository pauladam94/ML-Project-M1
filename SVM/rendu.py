import sys
import numpy as np
import gzip
from sklearn.cluster import KMeans
import pickle


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


def img2BW(img):
    x, y, c = img.shape
    img_ = np.zeros([x,y,1],dtype=np.int64)
    for i in range(x):
        for j in range(y):
            img_[i][j][0] = img[i][j][0]  + img[i][j][1] + img[i][j][2]
            img_[i][j][0] /= 3
    return(img_)


if len(sys.argv) != 2:
    print("Usage: python script.py <image_filename.csv.gz>\<label_filename.csv.gz>")
    sys.exit(1)

# Get filenames from command-line arguments
image_file = sys.argv[1]
#label_file = sys.argv[2]

# Load the images from the compressed CSV file
with gzip.open(image_file, 'rb') as f:
    data = np.loadtxt(f, delimiter=',').astype(np.int64)
    data = data.reshape(-1, 32, 32, 3)
#x_train = x_train.reshape(-1, 32, 32, 3)



data = [np.squeeze(kmeans2(convol(convol(convol(img2BW(image),ker3),ker5),ker3))) for image in data]

data = [image.flatten() for image in data]


with open('isThereOne.pkl', 'rb') as f:
    svm_model = pickle.load(f)


print(svm_model.predict(data))
