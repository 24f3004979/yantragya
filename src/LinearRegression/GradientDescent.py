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
    def __init__(self, training_dataset, seed_weight):
        '''
        training_dataset : {data values}
        Numpy_array with data with value pairs
        '''
        self.training_dataset = training_dataset
        self.seed_weight = np.array(seed_weight).reshape(-1,1)

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
                    cap_y = x.T @ self.seed_weight
                    print(f"inspection log : {cap_y.shape} | x : {x.shape} value: {x} with weight ; {self.seed_weight} with shape: {self.seed_weight.shape}")
                    predictions.append(cap_y)  # Final Prediction bundle
                except Exception as  e:
                    print(f"Computing cap_y terminated with :{x} with {self.seed_weight} {e}")
                    raise Exception(f"Terminating y_cap computation for graident ignition")
        return predictions

    def get_batch(self, array, batch_size=100):
        for i in range(0, len(array), batch_size):
            x_data = array[i:i+batch_size]
            x_data = preprocess(x_data)  # PREPROCESSING DATASET
            yield x_data

