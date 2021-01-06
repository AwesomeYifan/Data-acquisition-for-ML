from os import write
from random import sample
from Seller import Seller
import numpy as np
from RModel import RModel
from SPS import SPS_Buyer
import Utils
import random
import math
n_regions = 16
u = 1

(trainX, trainY), features, (testX, testY) = Utils.load_RoadNet()

folder = "roadnet/roadnet" + str(n_regions) + "/"
init_sample_ids = np.loadtxt(folder + str(u) + "/init.csv")
init_sample_ids = [int(v) for v in init_sample_ids]

seller = Seller(n_regions, trainX, trainY, False)
budget = 20000
for sz in [31]:
    # data acquisition
    write_file = open(folder + str(u) + "/SPS", 'w')
    for _ in range(10):
        seller.reset()
        ts_buyer = SPS_Buyer(budget, n_regions, features, trainX, trainY, init_sample_ids, seller)
        ts_buyer.chenge_to_regression()
        ts_buyer.batch_size = sz
        purchase_list = ts_buyer.process()
        write_file.write(' '.join([str(v) for v in purchase_list]) + "\n")
        write_file.flush()
    write_file.close()

    # model re-training
    write_file = open(folder + str(u) + "/results/SPS-sz" + str(sz) + "-results.csv", 'w')
    read_file = open(folder + str(u) + "/SPS", 'r')
    lines = read_file.readlines()
    for line in lines:
        more_sample_ids = line.split()
        more_sample_ids = [int(v) for v in more_sample_ids]
        for b in range(2000, len(more_sample_ids), 2000):
            image_ids = init_sample_ids.copy()
            image_ids.extend(more_sample_ids[:b])
            image_ids = list(dict.fromkeys(image_ids))
            print(str(b) + ": " + str(len(image_ids)))
            model = RModel()
            model.fit(trainX[image_ids], trainY[image_ids])
            score = model.score(testX, testY)
            write_file.write(str(b) + "," + str(score) + "\n")
            write_file.flush()
    write_file.close()