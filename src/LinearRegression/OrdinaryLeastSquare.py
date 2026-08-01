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
        '''
        Making dataset ready for further comptation

        Edit: Taking feature -> [feature] & target -> [target] | Now geting 5 X 5 as weight
        feature and feature naming convention crash

        reshaping fix into preprocessing could fix the downstream error
        '''

        features, target = self.np_convert(features, target)
        self.features = features  # Features defined

        self.X = self.preprocess(features)  # INFO: Taking feature into one more list :)
        self.w = None
        self.target = target

    def np_convert(self, *args):
        '''Simple conversion into tuples 
        Dictionary might break this function with value error'''

        converted = [np.array(lst) for lst in args]
        if len(args) == 1:
            return converted[0]  # First Unit
        return tuple(converted)  # Multiple Elements

    def preprocess(self, feature):
        '''Conversion into numpy object is also required'''
        feature = np.asarray(feature)

        if feature.ndim == 1:
            feature = feature.reshape(-1, 1)
            '''reshape-working
                Here It simply makes given number list of shape (feature_count, )
        With reshape(-1,1)
        compute number of rows required for making data into one column

            '''

        ones = np.ones((feature.shape[0], 1))

        stacked = np.hstack((ones, feature))
        return stacked

    def train_weight(self):
        '''
        Making simple computation for training for weights
        Improvements : require guarding rail terminations and error handlers
        '''
        try:
            if self.X.shape[0] != self.target.shape[0]:
                raise ValueError(f"Shape Missmatch feature shape : {self.X.shape} with target : {self.target}")
            pseudo_inverse = np.linalg.pinv(self.X)

            self.w = pseudo_inverse @ self.target
            return self.w  # weights for given dataset
        except Exception as e:
            raise Exception(f"Training terminated with {e}")

    def predict(self, new_data_point):
        if self.w is None:
            raise RuntimeError("Weights Not trained for predictions")
        new_X = self.preprocess(new_data_point)
        return new_X @ self.w

