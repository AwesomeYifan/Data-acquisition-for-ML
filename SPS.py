import numpy as np
from numpy.testing._private.utils import measure
from sklearn.metrics import pairwise_distances
from TS import TS
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import math
import Utils
import random
import Measure
class SPS_Buyer:
    def __init__(self, budget, n_regions, all_features, rawX, rawY, init_sample_ids, seller, model=0):
        self.partition_by_class = True
        self.budget = budget
        self.n_regions = n_regions
        self.all_features = all_features
        self.rawX = rawX
        self.rawY = rawY
        self.t_sampler = TS(n_regions)
        self.model = model
        self.seller = seller
        self.purchased_ids = []
        self.acquired_ids_by_region = [[] for i in range(n_regions)]
        self.init_sample_ids = init_sample_ids
        self.utility_list = [[] for i in range(n_regions)]
        self.all_utility_list = list()
        self.batch_size = 100
        self.tau = 1
        self.measure = 'novelty'
        self.retrain_enabled = False

    def process(self):
        self.initialize(self.init_sample_ids)
        # initialization
        for region_id in range(self.n_regions):
            sample_ids = self.seller.get_samples_of_region_id(region_id, self.batch_size)
            # print(len(self.purchased_ids))
            if len(sample_ids) == 0:
                self.t_sampler.set_empty(region_id)
                continue
            utility = self.compute_utility(sample_ids, region_id)
            self.t_sampler.update(region_id, utility, self.batch_size)
            self.merge_records(sample_ids, region_id)
            l = len(self.purchased_ids)
            if self.retrain_enabled:
                self.retrain()
        while len(self.purchased_ids) < self.budget:
            self.check_utility()
            print("*************************")
            region_id = self.t_sampler.next()
            sample_ids = self.seller.get_samples_of_region_id(region_id, self.batch_size)
            print(len(self.purchased_ids))
            if len(sample_ids) == 0:
                self.t_sampler.set_empty(region_id)
                continue
            utility = self.compute_utility(sample_ids, region_id)
            self.t_sampler.update(region_id, utility, self.batch_size)
            self.update_nonstationary_params(region_id)
            self.merge_records(sample_ids, region_id)
            if self.retrain_enabled:
                self.retrain()
        return self.purchased_ids
    
    def check_utility(self):
        for i in range(self.n_regions):
            print(str(len(self.acquired_ids_by_region[i])) + ": " + str(self.t_sampler.get_expected_reward(i)))
    
    def merge_records(self, sample_ids, region_id):
        self.purchased_ids.extend(sample_ids)
        new_sample_ids = list(dict.fromkeys([v for v in sample_ids if v not in self.init_sample_ids]))
        self.acquired_ids_by_region[region_id].extend(new_sample_ids)
       
    def initialize(self, init_sample_ids):
        if self.partition_by_class:
            for image_id in init_sample_ids:
                region_id = self.rawY[image_id]
                self.acquired_ids_by_region[region_id].append(image_id)
        else:
            for sample_id in init_sample_ids:
                region_id = Utils.get_region_id(self.all_features[sample_id], self.n_regions)
                self.acquired_ids_by_region[region_id].append(sample_id)
        
    def compute_utility(self, sample_ids, region_id):
        utility = 0
        if self.measure == 'novelty':
            utility = self.compute_novelty(sample_ids, region_id)
        elif self.measure == 'proxy-A-distance':
            utility = self.compute_proxy_dist(sample_ids, region_id)
        elif self.measure == 'uncertainty':
            utility = self.compute_uncertainty(sample_ids, region_id)
        elif self.measure == 'ACIMP':
            utility = self.compute_ACIMP(sample_ids)
        else:
            input('Invalid utility measure!')
        
        self.all_utility_list.append(utility)
        self.utility_list[region_id].append(utility)
        return utility
        
    def update_nonstationary_params(self, region_id):
        if len(self.utility_list[region_id]) > self.tau:
            reward_to_forget = self.utility_list[region_id][len(self.utility_list[region_id]) - self.tau - 1]
            self.t_sampler.forget(region_id, reward_to_forget, self.batch_size)

    def chenge_to_regression(self):
        self.partition_by_class = False
    
    def compute_uncertainty(self, sample_ids, region_id):
        testX = self.rawX[sample_ids]
        testY = np.array([region_id]*len(sample_ids))
        uncertainty = self.model.get_uncertainty(testX, testY)
        return uncertainty
    
    def compute_ACIMP(self, sample_ids):
        all_ids_before = self.init_sample_ids.copy()
        all_ids_before.extend(self.purchased_ids)
        all_ids_before = list(dict.fromkeys(all_ids_before))
        X = self.rawX[all_ids_before]
        y = self.rawY[all_ids_before]
        n_splits = 3
        score_before = 0
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X):
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            self.model.fit(train_X, train_y)
            score_before += self.model.score(test_X, test_y)
        score_before /= n_splits
        
        all_ids_after = all_ids_before
        all_ids_after.extend(sample_ids)
        all_ids_after = list(dict.fromkeys(all_ids_after))
        X = self.rawX[all_ids_after]
        y = self.rawY[all_ids_after]
        score_after = 0
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X):
            train_X, train_y = X[train_index], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            self.model.fit(train_X, train_y)
            score_after += self.model.score(test_X, test_y)
        score_after /= n_splits

        return max(score_after - score_before, 0)

    def compute_novelty(self, record_ids, label):
        old_vectors = self.get_acquired_vectors(label)
        new_vectors = self.all_features[record_ids]
        if not self.partition_by_class:
            new_vectors = []
            for record_id in record_ids:
                temp = self.all_features[record_id].tolist()
                temp.append(self.rawY[record_id])
                new_vectors.append(temp)
        acc = Measure.compute_novelty(old_vectors, new_vectors)
        return acc
   
    def compute_proxy_dist(self, record_ids, label):
        old_vectors = self.get_acquired_vectors(label)
        old_labels = [1]*len(old_vectors)
        new_vectors = self.all_features[record_ids]
        if not self.partition_by_class:
            new_vectors = []
            for record_id in record_ids:
                temp = self.all_features[record_id].tolist()
                temp.append(self.rawY[record_id])
                new_vectors.append(temp)
        new_labels = [0]*len(new_vectors)
        new_X_train = new_vectors
        new_y_train = new_labels
        X = list(new_X_train)
        X.extend(list(old_vectors))
        y = list(new_y_train)
        y.extend(list(old_labels))
        clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1).fit(X, y)
        acc = clf.score(X, y)
        return acc
    
    def get_acquired_vectors(self, region_id) :
        all_record_ids = self.acquired_ids_by_region[region_id]
        vectors = []
        if self.partition_by_class:
            for record_id in all_record_ids:
                vectors.append(self.all_features[record_id])
        else:
            for record_id in all_record_ids:
                vector = list(self.all_features[record_id].copy())
                vector.append(self.rawY[record_id])
                vectors.append(vector)
        return vectors
               
    def get_reward_trend(self):
        return self.all_utility_list

    def retrain(self):
        obtained_samples = self.init_sample_ids.copy()
        obtained_samples.extend(self.purchased_ids)
        obtained_samples = list(dict.fromkeys(obtained_samples))
        self.model.fit(self.rawX[obtained_samples], self.rawY[obtained_samples])