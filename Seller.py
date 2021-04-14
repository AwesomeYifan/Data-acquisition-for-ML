import Utils
import random
class Seller:
    def __init__(self, n_predicates, X, y, init_record_ids, partition_by_class = True):
        self.not_for_sale_enabled = True
        self.purchased_items_id = set(init_record_ids)
        self.n_predicates = n_predicates
        self.partition_by_class = partition_by_class
        self.data = self.build_data(X, y)
        self.sold = [[] for _ in range(len(self.data))]
        self.y = y
        

    def check_sizes(self):
        sizes = [len(item) for item in self.data]
        sizes.sort()
        print(sizes)
    
    def build_data(self, X, y):
        data = [[] for _ in range(self.n_predicates)]
        if self.partition_by_class:
            for i in range(len(X)):
                if self.not_for_sale_enabled and i in self.purchased_items_id:
                    continue
                data[y[i]].append((X[i], i))
        else:
            for i in range(len(X)):
                if self.not_for_sale_enabled and i in self.purchased_items_id:
                    continue
                region_id = Utils.get_region_id(X[i], self.n_predicates)
                data[region_id].append((X[i], i))

        return data
    
    def predicate_size(self, i):
        return len(self.data[i])


    def get_samples_of_region_id(self, label, amount):
        all_items = set([self.data[label][idx][1] for idx in range(len(self.data[label]))])
        all_items = all_items.difference(set(self.sold[label]))
        items_to_sell = list(all_items)
        if amount < len(all_items):
            items_to_sell = random.sample(all_items, amount)
        self.sold[label].extend(items_to_sell)
        return items_to_sell
    
    def is_empty(self, predicate_id):
        return len(self.sold[predicate_id]) == len(self.data[predicate_id])


    def reset(self):
        self.sold = [[] for _ in range(self.n_predicates)]