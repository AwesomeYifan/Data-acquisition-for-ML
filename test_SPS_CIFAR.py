
from Seller import Seller
import numpy as np
from skimage.feature import hog
from tensorflow.keras.datasets import cifar100, cifar10
from DModel import DModel
from tensorflow.keras.utils import to_categorical 
from SPS import SPS_Buyer
import random
import Utils
import math
dataset = "CIFAR10"
n_classes = 10

for u in [1]:
    
    init_image_ids = np.loadtxt("./" + dataset + "/" + str(u) + "/init.csv")
    init_image_ids = [int(v) for v in init_image_ids]
   
    (trainImages, trainLabels), features, (testX, testY) = Utils.load_CIFAR10()

    seller = Seller(n_classes, trainImages, trainLabels, init_image_ids)
    
    budget=39000

    # for sz in [100,200,300,400,500]:
    for sz in [301]:
        for tau in [1]:
            seller.reset()
            buyer = SPS_Buyer(budget, n_classes, features, trainImages, trainLabels, init_image_ids, seller)
            buyer.batch_size = sz
            buyer.tau = tau
            buyer.measure = 'novelty'
            purchase_list = buyer.process()
            for b in [3000,5000,6000,8000,10000,20000,30000,39000]:
            # for b in [5000,10000,20000]:
                # write_file = open("./" + dataset + "/" + str(u) + "/SPS" + "-" + str(b), 'w')
                write_file = open("./" + dataset + "/" + str(u) + "/SPS" + "-" + str(sz) + "-" + str(b), 'w')
                write_file.write(' '.join([str(v) for v in purchase_list[:b]]) + "\n")
                write_file.flush()
                write_file.close()
