import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
class RModel:
    def __init__(self):
        # self.model = MLPRegressor()
        # self.model = AdaBoostRegressor()
        self.model = KNeighborsRegressor()
    
    def fit(self, X, Y, verbosity=0):
        self.model.fit(X, Y)

    def score(self, testX, testY):
        r2 = self.model.score(testX, testY)
        r2 = max(r2, 1e-5)
        return r2