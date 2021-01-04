import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
class CModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()
        # self.model = KNeighborsClassifier()
        # self.model = RandomForestClassifier(n_estimators=100)
        # self.model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    
    def fit(self, X, Y):
        self.model.fit(X, Y)

    def score(self, testX, testY):
        acc = self.model.score(testX, testY)
        return acc
    
    def get_uncertainty(self, testX, testY):
        probs = self.model.predict_proba(testX)
        sum_uncertainty = 0
        for i in range(len(testX)):
            sum_uncertainty += 1 - probs[i, testY[i]]
        return sum_uncertainty / len(testX)
        
    def predict(self, testX):
        return self.model.predict(testX)