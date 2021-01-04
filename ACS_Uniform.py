
import random
import numpy as np
from CModel import CModel
from RModel import RModel
import Utils

# candidates = list(range(0,300000))
# for b in [2000,3000,5000,8000,10000,20000,30000,40000,50000]:
#     write_file = open("./CIFAR10/1/uniform-" + str(b), 'w')
#     for _ in range(1):
#         # write_file = open("./IUMP/IUMP16/0.002-0.2/uniform-" + str(b), 'w')
#         selected = random.sample(candidates, b)
#         # print(selected)
#         write_file.write(' '.join([str(v) for v in selected]))
#         write_file.flush()
#         write_file.close()

# roadnet
folder = './roadnet/roadnet16/1/'
(trainX, trainY), (testX, testY) = Utils.read_RoadNet()
candidates = list(range(0,len(trainX)))
write_file = open(folder + "uniform", 'w')
for _ in range(5):
    selected = random.sample(candidates, 300000)
    write_file.write(' '.join([str(v) for v in selected]) + "\n")
write_file.close()
init_image_ids = np.loadtxt(folder + "init.csv")
init_image_ids = [int(v) for v in init_image_ids]
write_file = open(folder + "results/uniform-results.csv", 'w')
read_file = open(folder + "uniform", 'r')
lines = read_file.readlines()
for line in lines:
    more_image_ids = line.split()
    more_image_ids = [int(v) for v in more_image_ids]
    for b in range(2000, len(more_image_ids), 2000):
        image_ids = init_image_ids.copy()
        image_ids.extend(more_image_ids[:b])
        image_ids = list(dict.fromkeys(image_ids))
        print(str(b) + ": " + str(len(image_ids)))
        model = RModel()
        model.fit(trainX[image_ids], trainY[image_ids])
        score = model.score(testX, testY)
        write_file.write(str(b) + "," + str(score) + "\n")
        write_file.flush()
write_file.close()