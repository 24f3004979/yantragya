'''
Implementing simple Linear regression
with simple pythonic implementation of pseudo inverse form
'''

import numpy as np

class OrdinaryLeastSquare:
    '''
    Ordinary way for computing weights with matrix format
    Input : 
    features, target
    
    train_weight : return -> weight

    input space is to be defined along with shape and data flow integrity

    '''
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.X = self.preprocess(feature)
        self.w = None # to train

    def preprocess(self, feature):
        ones = np.ones((feature.shape[0], 1))
        print(f"Horizontal Stacking unit : {ones}")

        stacked = np.hstack((ones, feature))
        print(f"Stacked Matrix : {stacked}")
        return stacked

    def train_weight(self):
        '''
        Making simple computation for training for weights
        Improvements : require guarding rail terminations and error handlers
        '''
        try:
            if self.X.shape[0] != self.target.shape[0]:
                raise ValueError(f"Shape Missmatch feature shape : {X.shape} with target : {self.target}")
            pseudo_inverse = np.linalg.pinv(self.X)
            self.w = pseudo_inverse @ y
            return self.w  # weights for given dataset
        except Exception as e:
            print(f"Terminating with Error {e}")
            raise Exception("Training terminated with {e}")

    def predict(self, new_data_point):
        if self.w is None:
            raise RuntimeError("Weights Not trained for predictions")
        new_X = self.preprocess(new_data_point)
        return new_X @ self.w


