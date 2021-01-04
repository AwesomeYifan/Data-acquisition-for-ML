import numpy as np
from numpy.core import records
from CModel import CModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import math
class AI_Buyer:
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
        self.batch_size = 100
    
    def process(self):
        X = self.rawX[self.init_sample_ids]
        y = self.rawY[self.init_sample_ids]
        accuracy_by_class_before = self.get_accuracy_by_class(X, y)
        portions = [1 / self.n_regions]*self.n_regions
        empty_classes = set()
        while self.remaining_budget > 0:
            print(len(self.acquired_ids))
            total_amount_to_acquire = min(self.remaining_budget, self.batch_size * self.n_regions)
            for i in range(self.n_regions):
                amt = int(math.ceil(total_amount_to_acquire * portions[i]))
                if amt == 0:
                    continue
                record_ids = self.seller.get_samples_of_region_id(i, amt)
                amt1 = len(record_ids)
                if len(record_ids) < amt:
                    empty_classes.add(i)
                self.remaining_budget -= len(record_ids)
                self.acquired_ids.extend(record_ids)
            all_record_ids = self.acquired_ids.copy()
            all_record_ids.extend(self.init_sample_ids)
            accuracy_by_class_after = self.get_accuracy_by_class(self.rawX[all_record_ids], self.rawY[all_record_ids])
            accuracy_diff = [accuracy_by_class_after[i] - accuracy_by_class_before[i] for i in range(self.n_regions)]
            accuracy_diff = [v if v > 0 else 0 for v in accuracy_diff]
            portions = [accuracy_diff[i] if i not in empty_classes else 0 for i in range(self.n_regions)]
            if len(empty_classes) == self.n_regions:
                break
            if sum(portions) == 0:
                portions = [1 / (self.n_regions - len(empty_classes)) if i not in empty_classes else 0 for i in range(self.n_regions)]
            portions = [v / sum(portions) for v in portions]
            accuracy_by_class_before = accuracy_by_class_after
        return self.acquired_ids

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