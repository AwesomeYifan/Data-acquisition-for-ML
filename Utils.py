import random
from time import sleep
import math
from skimage.feature import hog
from tensorflow.keras.datasets import cifar100, cifar10
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.manifold import TSNE
import statistics
import math
from sklearn import preprocessing
n_dims = 2

def load_CIFAR100():
    return load_CIFAR10('cifar100')

def load_CIFAR10(dataset='cifar10'):
    image_shape = (32, 32, 3)
    if dataset == 'cifar10':
        (trainX, trainY), (testX, testY) =  cifar10.load_data()
    elif dataset == 'cifar100':
        (trainX, trainY), (testX, testY) =  cifar100.load_data()
    else:
        print("specify a cifar dataset")
        return
    trainX, testX = prep_pixels(trainX, testX)
    trainY = trainY.flatten()
    testY = testY.flatten()
    # testY = to_categorical(testY, dtype ="uint8") 
    features = []
    for i in range(len(trainX)):
        fd = hog(trainX[i], multichannel = False if image_shape[2] == 1 else True)
        features.append(fd)
    trainX = trainX.reshape((trainX.shape[0], image_shape[0], image_shape[1], image_shape[2]))
    testX = testX.reshape((testX.shape[0], image_shape[0], image_shape[1], image_shape[2]))
    features = np.array(features)
    return (trainX, trainY), features, (testX, testY)

def load_Crop():
    data = np.loadtxt("./datasets/WinnipegDataset.txt", delimiter=',', skiprows=1)
    X = data[:,1:]
    y = data[:,0]
    y = np.array([int(v-1) for v in y])
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = min_max_scaler.fit_transform(X)
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, random_state=821)
    features = trainX.copy()
    return (trainX, trainY), features, (testX, testY)

def load_RoadNet():
    path = "./datasets/3D_spatial_network.txt"
    data = np.loadtxt(path, delimiter=",")
    X = data[:,1:3]
    y = data[:,3]
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X = min_max_scaler.fit_transform(X)
    y = min_max_scaler.fit_transform(y.reshape(-1,1))
    y = y.flatten()
    trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.2, random_state=821)
    features = trainX.copy()
    return (trainX, trainY), features, (testX, testY)

def prep_pixels(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm

def get_region_id(vector, n_regions):
    # n_dims = len(vector)
    n_segs_per_dim = int(n_regions ** (1 / n_dims))
    seg_length = 1.0 / n_segs_per_dim
    region_id = 0
    for i in range(n_dims):
        seg_id = int(math.floor(vector[i] / seg_length))
        seg_id = min(seg_id, n_segs_per_dim - 1)
        region_id += seg_id * int(pow(n_segs_per_dim, i))
    return region_id