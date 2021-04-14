import Measure
import Utils
import random
import numpy as np
from EA import EA_Buyer
from Seller import Seller

def partition_ids_roadnet(num_classes, X, ids):
    ids_by_class = [[] for _ in range(num_classes)]
    for id in ids:
        # class_label = labels[id]
        class_label = Utils.get_region_id(X[id], 16)
        ids_by_class[class_label].append(id)
    return ids_by_class

def partition_ids(num_classes, labels, ids):
    ids_by_class = [[] for _ in range(num_classes)]
    for id in ids:
        class_label = labels[id]
        ids_by_class[class_label].append(id)
    return ids_by_class

dataset = "CIFAR10"
n_classes = 10
(trainImages, trainLabels), features, (testX, testY) = Utils.load_CIFAR10()

init_image_ids = np.loadtxt("./" + dataset + "/1/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

remaining_ids = [i for i in range(len(features)) if i not in init_image_ids]

init_ids_by_class = partition_ids_roadnet(n_classes, trainLabels, init_image_ids)
remaining_ids_by_class = partition_ids_roadnet(n_classes, trainLabels, remaining_ids)
all_errors = []
for class_id in range(n_classes):

    all_samples_in_classes = features[remaining_ids_by_class[class_id]]
    possessed_samples = features[init_ids_by_class[class_id]]
    true_utility = Measure.compute_novelty(possessed_samples, all_samples_in_classes, len(all_samples_in_classes),10)
    print(true_utility)
    err_of_class = []
    for b in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]:
        budget = int(b * len(all_samples_in_classes))
        selected_ids = random.sample(remaining_ids_by_class[class_id], budget)
        acquired_samples = features[selected_ids]
        est_utility = Measure.compute_novelty(possessed_samples, acquired_samples, len(all_samples_in_classes), 10)
        err_of_class.append(abs(true_utility - est_utility))
        print(est_utility)
    all_errors.append(err_of_class)
    print("*************************")
all_errors = np.array(all_errors)
avg_errors = np.average(all_errors, axis=0)
print(avg_errors)

dataset = "CIFAR100"
n_classes = 100
(trainImages, trainLabels), features, (testX, testY) = Utils.load_CIFAR10()

init_image_ids = np.loadtxt("./" + dataset + "/1/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

remaining_ids = [i for i in range(len(features)) if i not in init_image_ids]

init_ids_by_class = partition_ids_roadnet(n_classes, trainLabels, init_image_ids)
remaining_ids_by_class = partition_ids_roadnet(n_classes, trainLabels, remaining_ids)
all_errors = []
for class_id in range(n_classes):

    all_samples_in_classes = features[remaining_ids_by_class[class_id]]
    possessed_samples = features[init_ids_by_class[class_id]]
    true_utility = Measure.compute_novelty(possessed_samples, all_samples_in_classes, len(all_samples_in_classes),6)
    print(true_utility)
    err_of_class = []
    for b in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]:
        budget = int(b * len(all_samples_in_classes))
        selected_ids = random.sample(remaining_ids_by_class[class_id], budget)
        acquired_samples = features[selected_ids]
        est_utility = Measure.compute_novelty(possessed_samples, acquired_samples, len(all_samples_in_classes), 6)
        err_of_class.append(abs(true_utility - est_utility))
        print(est_utility)
    all_errors.append(err_of_class)
    print("*************************")
all_errors = np.array(all_errors)
avg_errors = np.average(all_errors, axis=0)
print(avg_errors)

dataset = "roadnet"
n_classes = 16
(trainImages, trainLabels), features, (testX, testY) = Utils.load_RoadNet()
trainLabels = trainLabels.reshape((len(trainImages),1))
features = np.concatenate((features, trainLabels),axis=1)
init_image_ids = np.loadtxt("./" + dataset + "/roadnet16/1/init.csv")
init_image_ids = [int(v) for v in init_image_ids]

remaining_ids = [i for i in range(len(features)) if i not in init_image_ids]

init_ids_by_class = partition_ids_roadnet(n_classes, trainImages, init_image_ids)
remaining_ids_by_class = partition_ids_roadnet(n_classes, trainImages, remaining_ids)

all_errors = []
# for class_id in range(n_classes):
for class_id in [14]:
    all_samples_in_classes = features[remaining_ids_by_class[class_id]]
    if len(all_samples_in_classes) < 10:
        continue
    possessed_samples = features[init_ids_by_class[class_id]]
    true_utility = Measure.compute_novelty(possessed_samples, all_samples_in_classes, len(all_samples_in_classes))
    print(true_utility)
    err_of_class = []
    # for b in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]:
    for b in [0.1,0.2]:
        budget = int(b * len(all_samples_in_classes))
        selected_ids = random.sample(remaining_ids_by_class[class_id], budget)
        acquired_samples = features[selected_ids]
        est_utility = Measure.compute_novelty(possessed_samples, acquired_samples, len(all_samples_in_classes))
        err_of_class.append(abs(true_utility - est_utility))
        print(est_utility)
    all_errors.append(err_of_class)
    print("*************************")
all_errors = np.array(all_errors)
avg_errors = np.average(all_errors, axis=0)
print(avg_errors)