import Utils
import random
class Seller:
    def __init__(self, n_predicates, X, y, partition_by_class = True):
        self.n_predicates = n_predicates
        self.partition_by_class = partition_by_class
        self.data = self.build_data(X, y)
        self.sold = [[] for i in range(len(self.data))]
        self.purchased_items_id = []

    def check_sizes(self):
        sizes = [len(item) for item in self.data]
        sizes.sort()
        print(sizes)
    
    def build_data(self, X, y):
        data = [[] for i in range(self.n_predicates)]
        if self.partition_by_class:
            for i in range(len(X)):
                data[y[i]].append((X[i], i))
        else:
            for i in range(len(X)):
                region_id = Utils.get_region_id(X[i], self.n_predicates)
                data[region_id].append((X[i], i))

        return data
    
    def predicate_size(self, i):
        return len(self.data[i])

    def get_samples_of_region_id(self, label, amount):
        ori_list = list(range(len(self.data[label])))
        ori_list = list(set(ori_list) - set(self.sold[label]))
        if amount < len(ori_list):
            idxes = random.sample(range(0, len(ori_list)), amount)
            ori_list = [ori_list[i] for i in idxes]
        items_to_sell = [self.data[label][idx][1] for idx in ori_list]
        self.sold[label].extend(ori_list)
        return items_to_sell
    
    def is_empty(self, predicate_id):
        return len(self.sold[predicate_id]) == len(self.data[predicate_id])

    # def deliver(self):
    #     return self.purchased_items_id.copy()

    def reset(self):
        self.sold = [[] for i in range(self.n_predicates)]
        self.purchased_items_id = []