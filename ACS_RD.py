import numpy as np
from numpy.core import records
from CModel import CModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import random
import math
class RD_Buyer:
    def __init__(self, budget, n_regions, all_features, rawX, rawY, init_sample_ids, model, seller):
        self.partition_by_class = True
        self.budget = budget
        self.remaining_budget = budget
        self.n_regions = n_regions
        self.all_features = all_features
        self.init_sample_ids = init_sample_ids
        self.acquired_ids = []
        self.rawX = rawX
        self.rawY = rawY
        self.model = model
        self.seller = seller
        self.batch_size = 100 # to be adjusted by the test file
    
    def process(self):
        portions = [1 / self.n_regions]*self.n_regions
        empty_classes = set()
        while self.remaining_budget > 0:
            print(len(self.acquired_ids))
            total_amount_to_acquire = min(self.remaining_budget, self.batch_size * self.n_regions)
            ids_acquired_this_round = list()
            for i in range(self.n_regions):
                amt = int(math.ceil(total_amount_to_acquire * portions[i]))
                if amt == 0:
                    continue
                record_ids = self.seller.get_samples_of_region_id(i, amt)
                ids_acquired_this_round.extend(record_ids)
                amt1 = len(record_ids)
                if len(record_ids) < amt:
                    empty_classes.add(i)
                self.remaining_budget -= len(record_ids)
            rd_portion = self.get_rd_portion(ids_acquired_this_round)
            self.acquired_ids.extend(ids_acquired_this_round)
            portions = [rd_portion[i] if i not in empty_classes else 0 for i in range(self.n_regions)]
            if len(empty_classes) == self.n_regions:
                break
            if sum(portions) == 0:
                portions = [1 / (self.n_regions - len(empty_classes)) if i not in empty_classes else 0 for i in range(self.n_regions)]
            portions = [v / sum(portions) for v in portions]
        return self.acquired_ids
    
    def get_rd_portion(self, ids_acquired_this_round):
        rd_count = [0] * self.n_regions
        ids_acquired_before_this_round = self.init_sample_ids.copy()
        ids_acquired_before_this_round.extend(self.acquired_ids)
        random.shuffle(ids_acquired_before_this_round)
        random.shuffle(ids_acquired_this_round)
        X = self.rawX[ids_acquired_before_this_round]
        y = self.rawY[ids_acquired_before_this_round]
        n_splits = 5
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X):
            train_X, train_y = [X[i] for i in train_index], [y[i] for i in train_index]
            test_X, test_y = [X[i] for i in test_index], [y[i] for i in test_index]
            self.model.fit(train_X, train_y)
            labels_before = self.model.predict(test_X)
            new_X, new_y = [self.rawX[i] for i in ids_acquired_this_round], [self.rawY[i] for i in ids_acquired_this_round]
            train_X.extend(new_X)
            train_y.extend(new_y)
            self.model.fit(train_X, train_y)
            labels_after = self.model.predict(test_X)
            for i in range(len(test_X)):
                if labels_before[i] != labels_after[i]:
                    label = test_y[i]
                    rd_count[label] += 1
        n_records_per_class = [0]*self.n_regions
        for label in y:
            n_records_per_class[label] += 1
        rd_count = [rd_count[i] / n_records_per_class[i] for i in range(self.n_regions)]
        if sum(rd_count) == 0:
            rd_portion = [1 / self.n_regions] * self.n_regions
        else:
            rd_portion = [v / sum(rd_count) for v in rd_count]
        return rd_portion

    def get_accuracy_by_class(self, X, y):
        all_trues = []
        all_preds = []
        n_splits = 5
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(X):
            train_X, train_y = [X[i] for i in train_index], [y[i] for i in train_index]
            test_X, test_y = [X[i] for i in test_index], [y[i] for i in test_index]
            self.model.fit(train_X, train_y)
            all_preds.extend(self.model.predict(test_X))
            all_trues.extend(test_y)
        matrix = confusion_matrix(all_trues, all_preds)
        results = matrix.diagonal()/matrix.sum(axis=1)
        return results

    def chenge_to_regression(self):
        self.partition_by_class = False