from os import write
from Seller import Seller
import matplotlib.pyplot as plt
import numpy as np
from EA import EA_Buyer
import Utils
from RModel import RModel
import random
import Utils
u=1
n_regions = 16

(trainX, trainY), features, (testX, testY) = Utils.load_RoadNetd()

folder = "./roadnet/roadnet" + str(n_regions) + "/"
init_image_ids = np.loadtxt(folder + str(u) + "/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

all_budgets = list(range(2000,300000,2000))
alloc_strat = 'Linear'
seller = Seller(n_regions, trainX, trainY, False)

for l in [0.001]:
    # data acquisition
    for budget in all_budgets:
        write_file = open(folder + str(u) + "/EA-" + str(budget), 'w')
        for _ in range(10):
            seller.reset()
            buyer = EA_Buyer(budget, n_regions, features, trainX, trainY, init_image_ids, seller)
            buyer.chenge_to_regression()
            buyer.allocation_strategy = alloc_strat
            buyer.l = l
            purchase_list = buyer.process()
            write_file.write(' '.join([str(v) for v in purchase_list]))
            write_file.flush()
        write_file.close()

    # model re-training
    write_file = open(folder + str(u) + "/results/EA-" + alloc_strat + "-" + str(l) + "-results.csv", 'a')
    for budget in all_budgets:
        more_image_ids = np.loadtxt(folder + str(u) + "/EA-" + str(budget))
        more_image_ids = [int(v) for v in more_image_ids]
        image_ids = init_image_ids.copy()
        image_ids.extend(more_image_ids)
        image_ids = list(dict.fromkeys(image_ids))
        print(str(budget) + ": " + str(len(image_ids)))
        model = RModel()
        model.fit(trainX[image_ids], trainY[image_ids])
        score = model.score(testX, testY)
        write_file.write(str(budget) + "," + str(score) + "\n")
        write_file.flush()
    write_file.close()