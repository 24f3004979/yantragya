'''
Gradient Descent with Base ML
Batch processing for finding optimal weight
'''
from src.element.model import BaseML
import numpy as np


class GradientDescent(BaseML):
    '''
    parameter format
    epotch, batch_size, eta

    '''
    def __init__(self):
        super().__init__()

    def train(self, err_req=True):
        '''
        Epotch learning update rules
        with batch based processing to compute weights and update into gradient direction
        '''
        n = self.n# number of data point
        weight = np.array(self.hyper_para['initial_weight']).reshape(-1,1)
        epotches = self.hyper_para['epotches']
        batch_size = self.hyper_para['batch_size']
        eta = self.hyper_para['eta']

        err_list = [] # error listing
        iteration = 0 # Internal Iteration count log

        # Epotch Main Loop
        for epotch in range(epotches):
            indices = np.arange(n) # Index Shuffle
            np.random.shuffle(indices)
            # Shuffle Dataset for randomized approach
            X_shuffled = self.data[indices]
            y_shuffled = self.target[indices]
            
            for i in range(0, n, batch_size):
                # Batch for both target and labels
                x = X_shuffled[i: i+batch_size]
                y = y_shuffled[i: i+batch_size]

                pred = x.dot(weight)
                err = (pred - y)

                gradient = (2.0 / batch_size) * np.dot(x.T, err)

                weight = weight - (eta * gradient)

            # Global Error Analysis with current weight
            global_pred = self.data @ weight
            global_err = np.sum((global_pred - self.target)**2) / self.n 

            err_list.append(global_err) # Gloabl Error

            print(f"Gradient Descent Iteration : {iteration} of total {epotches} with Error : {global_err}")
            iteration += 1

        self.weights = weight # Final Weight Allication with training output
        if err_req:
            return err_list
