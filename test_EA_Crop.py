from os import write
from Seller import Seller
import matplotlib.pyplot as plt
import numpy as np
from EA import EA_Buyer
import Utils
from CModel import CModel
import random
import Utils
n_regions = 7
u = 5

(trainX, trainY), features, (testX, testY) = Utils.load_Crop()

init_image_ids = np.loadtxt("./Crop/" + str(u) + "/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

seller = Seller(n_regions, trainX, trainY)

all_budgets = list(range(2000,100001,5000))

alloc_strat = 'Linear'

for l in [0.01]:
    # data acquisition
    for budget in all_budgets:
        write_file = open("./Crop/" + str(u) + "/EA-" + alloc_strat + "-" + str(l) + "-" + str(budget), 'w')
        for _ in range(10):
            seller.reset()
            buyer = EA_Buyer(budget, n_regions, features, trainX, trainY, init_image_ids, seller)
            buyer.l=l
            buyer.allocation_strategy = alloc_strat
            purchase_list = buyer.process()
            write_file.write(' '.join([str(v) for v in purchase_list]) + "\n")
            write_file.flush()
        write_file.close()

    # model re-training
    write_file = open("./Crop/" + str(u) + "/results/EA-" + str(alloc_strat) + "-" + str(l) + "-results.csv", 'w')
    for budget in all_budgets:
        f = open("./Crop/" + str(u) + "/EA-" + alloc_strat + "-" + str(l) + "-" + str(budget), 'r')
        lines = f.readlines()
        for line in lines:
            more_image_ids = [int(v) for v in line.split()]
            image_ids = init_image_ids.copy()
            image_ids.extend(more_image_ids)
            image_ids = list(dict.fromkeys(image_ids))
            print(str(budget) + ": " + str(len(image_ids)))
            model = CModel()
            model.fit(trainX[image_ids], trainY[image_ids])
            score = model.score(testX, testY)
            write_file.write(str(budget) + "," + str(score) + "\n")
            write_file.flush()
    write_file.close()