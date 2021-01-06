from os import write
from Seller import Seller
import numpy as np
from skimage.feature import hog
from tensorflow.keras.datasets import cifar100, cifar10
import math
from DModel import DModel
from EA import EA_Buyer
import random
import Utils

dataset = "CIFAR10"
n_classes = 10
u = 1

(trainImages, trainLabels), features, (testX, testY) = Utils.load_CIFAR10()
init_image_ids = np.loadtxt("./" + dataset + "/" + str(u) + "/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

seller = Seller(n_classes, trainImages, trainLabels)

alloc_strat = 'Linear'
for budget in [3000,5000,8000,10000,20000,30000,40000,50000]:
    for l in [0.02]:
        seller.reset()
        buyer = EA_Buyer(budget, n_classes, features, trainImages, trainLabels, init_image_ids, seller)
        buyer.l = l
        buyer.allocation_strategy = alloc_strat
        purchase_list = buyer.process()
        write_file = open("./" + dataset + "/" + str(u) + "/EA-" + alloc_strat + "-" + str(l) + "-" + str(budget), 'w')
        write_file.write(' '.join([str(v) for v in purchase_list]))
        write_file.flush()
        write_file.close()