import Utils
import random
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
class TS:
    def __init__(self, num_regions):
        self.alphas = [1] * num_regions
        self.betas = [1] * num_regions
        self.is_empty = [0] * num_regions

    def next(self):
        max_theta = 0
        max_region_id = 0
        for i in range(len(self.alphas)):
            if self.is_empty[i]:
                continue
            theta = np.random.beta(self.alphas[i], self.betas[i])
            if theta > max_theta:
                max_theta = theta
                max_region_id = i
        return max_region_id
    
    def update(self, region_id, rwd, batch_size = 1):
        for _ in range(batch_size):
            self.alphas[region_id] += rwd
            self.betas[region_id] += (1 - rwd)
        # self.alphas[region_id] = max(self.alphas[region_id], 0.01)
        # self.betas[region_id] = max(self.betas[region_id], 0.01)
    
    def get_expected_reward(self, label):
        return self.alphas[label] / (self.alphas[label] + self.betas[label])
    
    def forget(self, region_id, rwd, batch_size = 1):
        
        self.alphas[region_id] -= rwd * batch_size
        self.betas[region_id] -= (1 - rwd) * batch_size
        # self.alphas[region_id] = max(self.alphas[region_id], 0.01)
        # self.betas[region_id] = max(self.betas[region_id], 0.01)
    
    
    def set_empty(self, region_id):
        self.is_empty[region_id] = 1

    def final_check(self):
        fig, ax = plt.subplots(1, 1)
        x = np.arange(0,1,0.01)
        for i in range(len(self.alphas)):
            print(str(self.alphas[i]) + "," + str(self.betas[i]))
            ax.plot(x, beta.pdf(x, self.alphas[i], self.betas[i]))
        plt.show()