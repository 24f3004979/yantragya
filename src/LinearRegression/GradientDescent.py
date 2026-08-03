'''
Gradient Descent algorithm
A sequential way for building weights for given dataset

Approach
1. Prediction with seed weight
    + computing y cap with seed weights for all training dataset
2. Gradient computation with respect to prediction
3. update logic with each iteration
'''
import numpy as np

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
                    print(f"Computed prediction : {cap_y}")
                    predictions.append(cap_y)  # Final Prediction bundle
                except Exception as  e:
                    print(f"Computing cap_y terminated with :{x} with {self.weight} {e}")
                    raise Exception(f"Terminating y_cap computation for graident ignition")
        return predictions

    def get_batch(self, array, batch_size=100):
        for i in range(0, len(array), batch_size):
            x_data = array[i:i+batch_size]
            yield x_data

    def train(self, epochs, neta):

        '''
        Running a Loop for epotch for training the weight with gradient based update
        neta : hyper parameter for the adjustment :)
        epotch: Convergence limit
        '''
        for epoch in range(epochs):
            predictions = self.y_cap(self.weight)
            diff = predictions - self.y
            loss = np.mean(diff ** 2)
            print(f"Gradient Inspection : {self.training_dataset.shape} with y :{self.y.shape} with diff {diff.shape}")
            gradient = (2 / len(self.y)) * self.training_dataset.T @ diff

            print(f"computed : {gradient}")
            self.weight -= neta * gradient

            print(f"Epoch {epoch}, Loss = {loss}")
