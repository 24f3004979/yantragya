'''
Gradient Descent algorithm
A sequential way for building weights for given dataset

Approach
1. Prediction with seed weight
    + computing y cap with seed weights for all training dataset
2. Gradient computation with respect to prediction
3. update logic with each iteration

---
# Implementation Documentation
current situation : it works, just works

Needs refinement about documenting each module working for and its use case into well documented plan for the given module to work, Thus no scalling or edits could be made precise into current situation

Problems
1. I am not clear about steps used into making this algorithm
2. No clarity about how really things are working with given  module
3. Data flow is not explained well thus can't get upgrade
4. No tests are currently made thus easy breakdown at load

Danger : Its not tested with good amount of dataset, and un-clear implementation makes it real danger to run with real big dataset, this would break and could crash pc also
'''
import numpy as np
import pandas as pd

# Preprocessing function imported from ordinary least square | Needs shift into utlility kit for usage
def preprocess(feature):
    feature = np.asarray(feature)

    if feature.ndim == 1:
        feature = feature.reshape(-1, 1)
        ones = np.ones((feature.shape[0], 1))

        stacked = np.hstack((ones, feature))
        return stacked


class GradientDescent:
    def __init__(self, training_dataset, weight):
        '''
        training_dataset : {data values}
        Numpy_array with data with value pairs

        critical constraints assumed in prototype
        X is the dataset column, y is target column 
        '''
        self.training_dataset = training_dataset["X"]
        self.y = training_dataset['y']

        # numpy array conversion for simple computation
        self.training_dataset = np.array(self.training_dataset).reshape(-1,1) # training Numbers
        self.y = np.array(self.y).reshape(-1,1)  # Target Numbers 
        self.weight = np.array(weight).reshape(-1,1)
        print(f"Initiating Gradient Descent with info | {self.weight} ")

    def y_cap(self, weight):
        '''
        Computing predictions with seed weights for given dataset
        working for making prediction
        '''
        print(f"starting with batch processing :) ")
        predictions = []
        for batch in self.get_batch(self.training_dataset):
            for x in batch:  # iterating fro the dataset [x:1]
                try:
                    # INFO: clarify shape transfer from this side
                    x = preprocess(x) # Now this should work :=)
                    print(f"Shape inspection with values : {x.shape} {x} with weight : {self.weight} with shape: {self.weight.shape}")
                    cap_y = x @ self.weight
                    print(f"Computed prediction : {cap_y.flatten()[0]}")
                    predictions.append(cap_y.flatten()[0])  # CRITICAL : due to matrix multiplication index used
                except Exception as  e:
                    print(f"Computing cap_y terminated with :{x} with {self.weight} {e}")
                    raise Exception(f"Terminating y_cap computation for graident ignition")
        # Shape fixes
        predictions = np.array(predictions)
        return predictions.reshape(-1,1)

    def get_batch(self, array, batch_size=100):
        for i in range(0, len(array), batch_size):
            x_data = array[i:i+batch_size]
            yield x_data

    def train(self, epochs, neta):
        weight = self.weight # Initial starting weights
        for epoch in range(epochs):
            pred = self.y_cap(weight)
            dif = (pred - self.y) # Difference vector for given weight

            product = self.training_dataset.T @ dif # Matrix multiplication
            product = product.flatten()[0]
            gradient = (2/self.training_dataset.size) * product

            weight = weight - neta * gradient
            print(f"Epotch {epoch} with Loss value : {np.average(dif)}")
        print(f'Final Weight computed : {weight} with loss {np.average(dif)}')
