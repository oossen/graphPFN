import numpy as np


class PFNWrapper:
    def __init__(self, reg, prob_adj=None):
        self.reg = reg
        self.prob_adj = prob_adj
    
    def fit(self, X_train, y_train):
        self.reg.fit(X_train.to_numpy(), y_train.to_numpy())
        
    def predict(self, X_test):
        if self.prob_adj is None:
            preds = self.reg.predict(X_test.to_numpy())
        else:
            preds = self.reg.predict(X_test.to_numpy(), prob_adj=self.prob_adj)
        return preds