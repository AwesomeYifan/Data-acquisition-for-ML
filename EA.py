from random import sample
import numpy as np
from numpy.lib import utils
from sklearn.metrics import pairwise_distances
from sklearn.utils import class_weight
from TS import TS
from scipy.stats import t
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import math
import Utils
import random
class Statistical_Buyer:
    def __init__(self, budget, n_predicates, all_features, rawX, rawY, init_record_ids, model, seller):
        self.partition_by_class = True
        self.budget = budget
        self.remaining_budget = budget
        self.n_predicates = n_predicates
        self.all_features = all_features # HOG features of images
        self.rawX = rawX
        self.rawY = rawY
        self.init_record_ids = init_record_ids
        self.model = model
        self.seller = seller
        self.acquired_record_ids_by_predicate = [[] for _ in range(n_predicates)]
        self.initial_vectors_by_class = []
        self.utility_list = [0]*n_predicates
        # self.num_interactions = 0
        self.delta = 1e-3
        self.l = 0.01
        self.allocation_strategy = 'Squareroot' # or 'Linear'

    def process(self):
        self.initial_vectors_by_class = self.get_init_vectors(self.rawY, self.init_record_ids)
        self.initialize()
        estimated_utilities = self.probe()
        print("Budget consumption for eatimation: " + str(self.budget - self.remaining_budget))
        # for i in range(self.n_predicates):
        #     percentage = estimated_utilities[i] / sum(estimated_utilities)
        #     percentage = int(percentage * 100)/100
        #     # amount = int(estimated_utilities[i] / sum(estimated_utilities) * self.budget)
        #     print(str(i) + ": " + str(percentage), end=";")
        # input("done")
        self.allocate(estimated_utilities)
        purchased_ids = []
        for l in self.acquired_record_ids_by_predicate:
            purchased_ids.extend(l)
        # print("num. interactions: " + str(self.num_interactions))
        return purchased_ids

    def probe(self):
        while True:
            # self.utility_list = [1]*self.n_predicates
            current_eps = self.get_current_eps()
            if current_eps == 0 or self.remaining_budget == 0:
                break
            current_PG = self.get_heuristic_PG(self.remaining_budget, current_eps)
            max_possible_PG, delta_amounts = self.decide(current_eps)
            if max_possible_PG <= current_PG or sum(delta_amounts) == 0:
                break
            # self.num_interactions+=1
            for i in range(len(delta_amounts)):
                if delta_amounts[i] == 0:
                    continue
                record_ids = self.seller.get_samples_of_region_id(i, delta_amounts[i])
                self.remaining_budget -= len(record_ids)
                self.acquired_record_ids_by_predicate[i].extend(record_ids)
                # self.uncertainty_list[i] = self.compute_uncertainty(i)
                self.utility_list[i] = self.compute_utility(i)
        estimated_utilities = self.utility_list
        return estimated_utilities
    
    def decide(self, current_eps):
        min_ps = []
        for i in range(self.n_predicates):
            p = self.utility_list[i]
            min_ps.append(p)
        # best_eps = 0
        max_PG = 0
        best_delta_amounts = []
        for eps in np.arange(current_eps*0.99, 0, -current_eps*0.01):
            delta_amounts = [0] * self.n_predicates
            for i in range(self.n_predicates):
                if len(self.acquired_record_ids_by_predicate[i]) <= 1:
                    continue
                ap0 = -t.ppf(q=self.delta/2,df=len(self.acquired_record_ids_by_predicate[i])-1)
                sp0 = min_ps[i] * (1-min_ps[i])
                delta_amount = (ap0 * sp0 / eps) ** 2 - len(self.acquired_record_ids_by_predicate[i])
                if delta_amount <= 0:
                    continue
                delta_amount = int(math.ceil(delta_amount))
                delta_amounts[i] = delta_amount
            PG = self.get_heuristic_PG(self.remaining_budget - sum(delta_amounts), eps)
            if PG > max_PG:
                best_delta_amounts = delta_amounts
                max_PG = PG
        return max_PG, best_delta_amounts

    def initialize(self):
        # self.num_interactions+=1
        for i in range(self.n_predicates):
            amount = max(5, int(math.ceil(self.l*self.seller.predicate_size(i))))
            record_ids = self.seller.get_samples_of_region_id(i, amount)
            if len(record_ids) == 0:
                continue
            self.remaining_budget -= len(record_ids)
            self.acquired_record_ids_by_predicate[i].extend(record_ids)
            utility = self.compute_utility(i)
            self.utility_list[i] = utility
       
    def get_current_eps(self):
        eps = 0
        for i in range(self.n_predicates):
            if len(self.acquired_record_ids_by_predicate[i]) == 0:
                continue
            p = self.utility_list[i]
            sp = p * (1-p)
            n_acquired_records = len(self.acquired_record_ids_by_predicate[i])
            zp = -t.ppf(q=self.delta/2, df = n_acquired_records-1)
            this_eps = zp * sp / math.sqrt(n_acquired_records)
            eps = max(eps, this_eps)
        return eps

    def get_heuristic_PG(self, remaining_budget, eps):
        # return np.exp(-eps) * remaining_budget / self.budget
        return (1-eps) * remaining_budget / self.budget # divide by self.budget is just to normalize

    def allocate(self, estimated_utilities):
        estimated_utilities = [v if v > 0 else 1e-5 for v in estimated_utilities]
        # in case all predicates with utility>0 are exhausted
        if self.allocation_strategy == 'Linear':
            estimated_utilities = estimated_utilities
        elif self.allocation_strategy == 'Square':
            estimated_utilities = [v**2 for v in estimated_utilities]
        elif self.allocation_strategy == 'Squareroot':
            estimated_utilities = [v**0.5 for v in estimated_utilities]
        elif self.allocation_strategy == 'Log':
            estimated_utilities = [np.exp(1-v) for v in estimated_utilities]
        else:
            input("invalid allocation strategy!")
        # first round
        utility_sum = sum(estimated_utilities)
        unit_amount = self.budget / utility_sum
        for i in range(self.n_predicates):
            amount = min(int(unit_amount * estimated_utilities[i]),self.remaining_budget)
            amount_to_buy = amount - len(self.acquired_record_ids_by_predicate[i])
            if amount_to_buy <= 0:
                continue
            record_ids = self.seller.get_samples_of_region_id(i, amount_to_buy)
            self.remaining_budget -= len(record_ids)
            self.acquired_record_ids_by_predicate[i].extend(record_ids)
        # if budget is not exhausted
        while self.remaining_budget > 0:
            utility_sum = 0
            for i in range(self.n_predicates):
                if self.seller.is_empty(i):
                    continue
                utility_sum += estimated_utilities[i]
            unit_amount = int(math.ceil(self.remaining_budget / utility_sum))
            for i in range(self.n_predicates):
                if self.seller.is_empty(i):
                    continue
                amount_to_buy = min(int(math.ceil(unit_amount * estimated_utilities[i])),self.remaining_budget)
                record_ids = self.seller.get_samples_of_region_id(i, amount_to_buy)
                self.remaining_budget -= len(record_ids)
                self.acquired_record_ids_by_predicate[i].extend(record_ids)
            

    def get_init_vectors(self, rawY, init_record_ids):
        vectors_obtained = [[] for i in range(self.n_predicates)]
        if self.partition_by_class:
            for sample_id in init_record_ids:
                label = rawY[sample_id]
                vectors_obtained[label].append(self.all_features[sample_id])
        else:
            for sample_id in init_record_ids:
                region_id = Utils.get_region_id(self.all_features[sample_id], self.n_predicates)
                vector = list(self.all_features[sample_id]).copy()
                vector.append(self.rawY[sample_id])
                vectors_obtained[region_id].append(vector)
        return vectors_obtained

    def compute_utility(self, label):
        old_vectors = self.initial_vectors_by_class[label]
        old_vectors = np.array(old_vectors)
        old_labels = [1]*len(old_vectors)
        record_ids = self.acquired_record_ids_by_predicate[label]
        new_vectors = self.all_features[record_ids]
        if not self.partition_by_class:
            append_values = np.array([self.rawY[record_ids]]).T
            new_vectors = np.concatenate((new_vectors, append_values),axis=1)
        new_labels = [0]*len(new_vectors)
        new_X_train = new_vectors
        new_X_test = new_vectors
        new_y_train = new_labels
        new_y_test = new_labels
        X = list(new_X_train)
        X.extend(list(old_vectors))
        y = list(new_y_train)
        y.extend(list(old_labels))
        # X, X_test, y, y_test = train_test_split(X, y, stratify=y, random_state=1)
        clf = KNeighborsClassifier(n_neighbors=3).fit(X, y)
        # clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)
        # clf = DecisionTreeClassifier().fit(X, y)
        acc = clf.score(new_X_test, new_y_test)
        return acc
    # def compute_utility(self, label):
    #     old_vectors = self.initial_vectors_by_class[label]
    #     # old_vectors = np.array(old_vectors)
    #     old_labels = [1]*len(old_vectors)
    #     record_ids = self.acquired_record_ids_by_predicate[label]
    #     new_vectors = self.all_features[record_ids]
    #     if not self.partition_by_class:
    #         new_vectors = []
    #         for record_id in record_ids:
    #             temp = self.all_features[record_id].copy().append(self.rawY[record_id])
    #             new_vectors.append(temp)
    #     new_labels = [0]*len(new_vectors)
    #     n_splits = 5
    #     sum_acc = 0
    #     kf = KFold(n_splits=n_splits)
    #     for train_index, test_index in kf.split(new_vectors):
    #         new_X_train, new_y_train = [new_vectors[i] for i in train_index], [new_labels[i] for i in train_index]
    #         new_X_test, new_y_test = [new_vectors[i] for i in test_index], [new_labels[i] for i in test_index]
    #         X = new_X_train.copy()
    #         X.extend(list(old_vectors))
    #         y = new_y_train.copy()
    #         y.extend(list(old_labels))
    #         clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
    #         acc = clf.score(new_X_test, new_y_test)
    #         sum_acc += acc
    #     return sum_acc / n_splits
    
    def chenge_to_regression(self):
        self.partition_by_class = False