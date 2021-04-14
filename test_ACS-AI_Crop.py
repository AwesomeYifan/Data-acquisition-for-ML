from os import write
from Seller import Seller
import numpy as np
from CModel import CModel
from ACS_AI import AI_Buyer
import Utils
import math
import random

n_classes = 7
folder = "./Crop/1/"
def load_dataset():
    (trainX, trainY), (testX, testY) =  Utils.read_Crop()
    features = trainX.copy()
    return (trainX, trainY), features, (testX, testY)
 
(trainX, trainY), features, (testX, testY) = load_dataset()

init_image_ids = np.loadtxt(folder + "init.csv")
init_image_ids = [int(v) for v in init_image_ids]

total_budget = 100001
budget_list = list(np.arange(2000,100001,5000))

seller = Seller(n_classes, trainX, trainY, init_image_ids)
# purchase step

for _ in range(10):
    seller.reset()
    model = CModel()
    buyer = AI_Buyer(total_budget, n_classes, features, trainX, trainY, init_image_ids, model, seller)
    buyer.batch_size = 100
    purchase_list = buyer.process()
    for budget in budget_list:
        write_file = open(folder + "AI-" + str(budget), 'a')
        write_file.write(' '.join([str(v) for v in purchase_list[:budget]]) + "\n")
        write_file.flush()
        write_file.close()

# evaluation step
write_file = open(folder + "results/AI-results.csv", 'w')
for budget in budget_list:
    model = CModel()
    f = open(folder + "AI-" + str(budget), 'r')
    lines = f.readlines()
    for line in lines:
        more_record_ids = [int(v) for v in line.split()]
        record_ids = init_image_ids.copy()
        record_ids.extend(more_record_ids)
        record_ids = list(dict.fromkeys(record_ids))
        model = CModel()
        model.fit(trainX[record_ids], trainY[record_ids])
        score = model.score(testX, testY)
        write_file.write(str(budget) + "," + str(score) + "\n")
        write_file.flush()
write_file.close()