'''
Gradient descent Model
Inheritence of RootML class for core model handle
'''
from src.element.modelv2 import RootML
import numpy as np


class GradientDescent(RootML):
    '''
    Gradient Descent Algorithm Unit
    After model initiation load dataset into model instance
    hyper_parameters = {
        eta : 0.0001, # learning rate
        batch_size : 20, # training batch
        initiating_weight : np.array[3,3] # must be vector with (d,1) shape
        epotches : 20
    }
    '''
    def __init__(self):
        super().__init__()
        self.error_history = []  # Error History

    def train(self):
        '''
        Training logic
            Iterating batches for dataset
            gradient computatioon
            recursive addition
        '''
        # Initiating variables
        epotches = self.hyper_parameters.get('epotches')
        eta = self.hyper_parameters.get('eta')
        batch_size = self.hyper_parameters.get('batch_size')
        initiating_weight = self.hyper_parameters.get('initiating_weight')

        n,d = self.train_X.shape
        test_n = len(self.test_X)
        self.weights = np.array(initiating_weight).reshape(-1,1)

        for epotch in range(epotches):

            for i in range(0, n, batch_size):
                x = self.train_X[i:i+batch_size]
                y = self.train_y[i:i+batch_size]

                prediction = x.dot(self.weights)
                error = (prediction - y)

                gradient = (2/batch_size) * np.dot(x.T, error)
                self.weights = self.weights - (eta * gradient)

            # Epotch directional testing plugin
            global_prediction = self.predict(self.test_X)
            global_error = (np.sum((global_prediction - self.test_y)**2) / test_n)
            self.error_history.append(global_error)
            print(f"Epotch :{epotch} with Erorr : {global_error}")
