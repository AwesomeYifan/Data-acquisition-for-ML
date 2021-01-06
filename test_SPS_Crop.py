from os import write
from Seller import Seller
import numpy as np
from CModel import CModel
from SPS import SPS_Buyer
import Utils
import math
import random
n_classes = 7
u = 1

(trainX, trainY), features, (testX, testY) = Utils.load_Crop()

init_image_ids = np.loadtxt("./Crop/" + str(u) + "/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

model = CModel()
model.fit(trainX[init_image_ids], trainY[init_image_ids])
seller = Seller(n_classes, trainX, trainY)
budget = 100001
for sz in [100]:
    # data acquisition
    write_file = open("./Crop/" + str(u) + "/SPS", 'w')
    for _ in range(10):
        seller.reset()
        buyer = SPS_Buyer(budget, n_classes, features, trainX, trainY, init_image_ids, seller, model)
        buyer.batch_size = sz
        buyer.measure = 'novelty'
        purchase_list = buyer.process()
        write_file.write(' '.join([str(v) for v in purchase_list]) + "\n")
        write_file.flush()
    write_file.close()

    # model re-trainings
    write_file = open("./Crop/" + str(u) + "/results/ts-novelty-results.csv", 'w')
    f = open("./Crop/" + str(u) + "/SPS", 'r')
    lines = f.readlines()
    for line in lines:
        more_image_ids = [int(v) for v in line.split()]
        for b in range(2000,100001,5000):
            for _ in range(10):
                image_ids = init_image_ids.copy()
                image_ids.extend(more_image_ids[:b])
                image_ids = list(dict.fromkeys(image_ids))
                print(str(b) + ": " + str(len(image_ids)))
                model = CModel()
                model.fit(trainX[image_ids], trainY[image_ids])
                score = model.score(testX, testY)
                write_file.write(str(b) + "," + str(score) + "\n")
                write_file.flush()
    write_file.close()