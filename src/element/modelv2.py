import numpy as np 
from src.util.logger import log
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

'''
RootML contract statements
1. Dataset must be into (n,d) shape
    Due to less flexible code base all derivations are aligned with this convention
2. Dont pass with null values | Strict Data check is must
    Null in dataset can derive whole pipeline infection
3. Loaded data is expected to be numpy array | optional
4. Intrcept is loaded with (data, ones)
    Following last weight vector would be the intercept element
'''

class RootML(ABC):
    '''
    Fundamental object for ml models

    parameters
        - weights 
        - data
        - target
        - hyper_parameters
    Functions
        - prediction
        - loading dataset
        - shape config
        - train | abstract method
        - Analysis of model
        - train/test split logic simplified 
    '''
    def __init__(self) -> None:
        self.hyper_parameters = None
        self.weights = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

    def preprocess(self,data, intercept_required=True):
        '''
        Basic preprocessing unit

        functions
        1. numpy data conversion
        2. shape transform [ trim - cut - consistent shape ]
        3. test prediction pipeline
        4. Load Normalization if defined [ future upgrade ]
        '''
        data = np.array(data)  # Numpy conversion
        n = data.shape[0] # number of data points

        if data.ndim == 1:  # conditioned for flat dataset
            data = data.reshape(-1,1)

        if intercept_required:
            ones = np.ones((n,1))  # shape (n,1) -> stack into data
            intercept_dataset = np.hstack((data, ones))
            log.info(f"RootML.preprocess | Intercept dataset : \n {intercept_dataset} \n")

            return intercept_dataset
        return data

    def load(self, data, target,split_ratio=(80,20), hyper_parameters=None):
        '''
        Loading dataset
        train-test split
        internal parameter activation
        '''
        if not(hyper_parameters is None):
            self.hyper_parameters = hyper_parameters  # loading hyper parameters
        indices_shuffle = np.random.permutation(len(data))  # index reference

        # Numpy conversions
        data = np.array(data)
        target = self.preprocess(target, intercept_required=False)

        # Shuffle Dataset
        shuffle_data = data[indices_shuffle]
        shuffle_target = target[indices_shuffle]

        # split
        split_index = int(len(data) * 0.8)

        train_X, self.test_X = np.split(shuffle_data, [split_index])
        train_y, test_y = np.split(shuffle_target, [split_index])

        self.train_y = train_y.flatten()
        self.test_y = test_y.flatten()
        

        self.train_X = self.preprocess(train_X)  # intercept injected

        log.info(f'Train Test spiliting output : {self.train_X.shape} | {self.train_y.shape}')

    @abstractmethod
    def train(self):
        '''Core training function'''
        pass

    def predict(self, new_data_point):
        testing_point = self.preprocess(new_data_point)

        if self.weights is None:
            raise Exception("Model is not trained | terminating prediction")
        if self.weights.shape[0] != testing_point.shape[1]:
            raise Exception(f"weights missmatch weights : {self.weights.shape} with testing point : {testing_point.shape}")

        log.info(f"Prediction Generation with : testing point : {testing_point.shape} and weights : {self.weights.shape}")
        return testing_point @ self.weights  # prediction points

    def visualize(self):
        '''
        Making plot about model performance with given matrices
        visualizing train, tested prediction with plots
        
        Visualization sequencing for the dataset, 
        Loading root train and test splits into plot for visualization

        We would initiate making visualization tool for this in future
        '''
        pass
        