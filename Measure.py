from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KernelDensity
import math
from pyod.models.vae import VAE
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import random
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

def compute_novelty(X_possess, X_acquire_total, length=-1, scale_factor=1):
    
    if len(X_acquire_total) < 5:
        return 0
    scale_factor = 1 # 10 for CIFAR10, 6 for CIFAR100, 1 for Crop and RoadNet
    total_acc = 0
    times = 50
    for _ in range(times):
        y_new = [1]*len(X_acquire_total)
        X_acquire, X_new_test, _, y_new_test = train_test_split(X_acquire_total, y_new, test_size = 0.2)
        X_new = X_acquire
        if len(X_acquire_total) < length:
            # no need to scale in practice for efficiency consideration
            # scale is just used to measure how accurate the estimation is
            mid_length = len(X_acquire) + int((length*0.8 - len(X_acquire)) / scale_factor)
            
            y_new = [1] * len(X_new)
            X_assist = [[random.uniform(0,1) for _ in range(len(X_new[0]))] for _ in range(mid_length)]
            y_assist = [0] * len(X_assist)
            X = X_assist
            X.extend(list(X_new))
            y = y_assist
            y.extend(list(y_new))
            oversample = SMOTE(k_neighbors=len(y_new)-1)
            X, y = oversample.fit_resample(X, y)
            X_new = []
            for i in range(len(y)):
                if y[i] == 1:
                    X_new.append(list(X[i]))
            X_new = resample(X_new, n_samples = int(length*0.8))
        X_old = X_possess
        
        y_old = [0]*len(X_old)
        y_new = [1]*len(X_new)
        X_new_train = X_new
        y_new_train = y_new
        X = list(X_new_train)
        X.extend(list(X_old))
        y = list(y_new_train)
        y.extend(list(y_old))
        
        k = 1
        clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        # clf = DecisionTreeClassifier().fit(X, y)
        # clf = Perceptron().fit(X, y)
        acc = clf.score(X_new_test, y_new_test)
        total_acc += acc
    return total_acc / times
