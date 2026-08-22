'''
OLS
Findinng Best fit with straight Linear Algebra one shot
for more information about OLS visit documentation under docs/OLS.md
'''
import numpy as np
from src.util.logger import log

class OLS:
    def __init__(self, X, target,feat_num=1):
        '''
        X requirements
        number of features is must required for the transformations
        np matrix of dataset
        Shape requirements
        X ~ (n,d)
        '''
        self.X = X
        self.target = target
        self.weights = None
        self.feat_num = feat_num

    def train(self):
        X = np.array(self.X)
        y = np.array(self.target)
        
        # Avoid shape missmatch
        X = X.reshape(-1, self.feat_num)

        sample = X.shape[0]
        ones_column = np.ones((sample, 1))
        
        # stacking for intercept
        print(f'shape inspection : {ones_column.shape} with {X.shape}')
        x = np.hstack([ones_column, X])

        p_inv = np.linalg.pinv((x.T).dot(x))
        x_y = (x.T).dot(y)

        log.info(f"Inverse computed : {p_inv.shape} \n {p_inv}")

        self.weights = p_inv.dot(x_y)  # Final computation
        log.info(f"Weights computed : {self.weights}")
        return self.weights

    def predict(self, X_new):
        X_new = np.array(X_new)
        X_new = X_new.reshape(-1, self.feat_num)
        sample = X_new.shape[0]
        ones_column = np.ones((sample, 1))

        x = np.hstack([ones_column, X_new])

        if x.shape[-1] != self.weights.shape[0]:
            raise Exception('Shape Miss Match for prediction')
            
        print(f"Inspecting for Nans : {x} with {self.weights}")
        return x @ self.weights



