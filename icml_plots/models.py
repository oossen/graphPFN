import numpy as np


class PFNWrapper:
    def __init__(self, reg, prob_adj=None):
        self.reg = reg
        self.prob_adj = prob_adj
    
    def fit(self, X_train, y_train):
        self.numeric_cols = X_train.select_dtypes(include=['number']).columns
        self.train_mean_X = X_train[self.numeric_cols].mean()
        self.train_std_X = X_train[self.numeric_cols].std()
        # clip std to avoid division by zero
        self.train_std_X = np.maximum(self.train_std_X, 1)
        X_train_scaled = X_train.copy()
        X_train_scaled[self.numeric_cols] = (X_train[self.numeric_cols] - self.train_mean_X) / self.train_std_X

        self.train_mean_y = y_train.mean()
        self.train_std_y = y_train.std()
        # clip std to avoid division by zero
        self.train_std_y = np.maximum(self.train_std_y, 1)
        y_train_scaled = (y_train - self.train_mean_y) / self.train_std_y
        
        self.reg.fit(X_train_scaled.to_numpy(), y_train_scaled.to_numpy())
        
    def predict(self, X_test):
        X_test_scaled = X_test.copy()
        X_test_scaled[self.numeric_cols] = (X_test[self.numeric_cols] - self.train_mean_X) / self.train_std_X
        
        if self.prob_adj is None:
            preds_scaled = self.reg.predict(X_test_scaled.to_numpy())
        else:
            preds_scaled = self.reg.predict(X_test_scaled.to_numpy(), prob_adj=self.prob_adj)
        preds_final = (preds_scaled * self.train_std_y) + self.train_mean_y
        
        return preds_final