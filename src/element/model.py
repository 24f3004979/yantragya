'''
Defining core model object for all models
An root inheritence object for all models to work with

Functionality and usecase
1. Loading dataset
2. Sanitization for model computation
    data cleaning is loaded in external script
3. Overriding options for models
4. Documented structure for easy refrences
5. Foundational requirements

----
Assumptions and Rules

shape is considered as (n,d) for core data module
    Must be followed during loadig into the feature space

BUG:There are some critical issues with the core model | Needs validation and refactor :)

'''
import numpy as np
from src.util.logger import log
from abc import ABC, abstractmethod


class BaseML(ABC):
    '''
    Core Root ML object

    parameters
        - weight
        - data_points [ shape ]
        - core training loop
        - hyper_parameter dictionary
        - dedicated prediction function
        - sanitization for preparation of data loading
    Features
        - Load dataset
            Initiates internal data objects and loads shape configs
        - train
            Over riding function for  all models
            with their core training logic
        - preprocess
            data loading routines for easy load
        - predict[input_dataset]
            Makes prebuilt config loaded data preparation
            One unified structure for all
            optional override for dedicated modules
    '''
    def __init__(self):
        self.hyper_para = None
        self.weights = None
        self.data = None
        self.target = None

        # Shape config requirements
        self.n = 0 # number of points
        self.d = 0 # features

    def preprocess(self, data, intercept_required=True):
        '''
        Making itercepting column addition to dataset parameter controlled
        Loading into numpy object
        shape check - exception fail
        '''
        data = np.array(data)  # Loading with Numpy requirements

        if data.ndim == 1:
            data = data.reshape(-1,1)
        
        # Saving to add interception into Wrong dataset 
        if self.d != 0:
            if data.shape[1] != self.d:  # Raw data dimensions are checked here
                log.warning(f"Terminating Preprocess with shape crash : {data.shape} with {self.weights.shape}")
                #raise Exception("Testing Data is Not in shape convention | terminating preprocess")

        n = data.shape[0]
        if intercept_required:
            ones = np.ones((n,1))
            data = np.hstack((data, ones))
            log.info(f"Preprocess : Model Intercept Added Dataset {data.shape}")
            return data
        log.warning(f"Preprocess : JUST LOADED DATA WITH NUMPY WITHOUT INTERCEPT")
        return data

    def load(self, Dataset, target, hyper_para):
        '''
        Input requirements
        dataset with shape (n,d) data_samples with feature
        hyper_para = {
            'alpha' : value,
            'beta' : value,
            'gama' : value
        }
        INFO : Target is strictly taken into with (n,1) for simplicity
        '''
        # Target special config
        target = np.array(target).reshape(-1,1)
        self.data = self.preprocess(Dataset)

        self.n, self.d = self.data.shape
        self.target = target
        self.hyper_para = hyper_para # used into training Model

    @abstractmethod
    def train(self):
        '''
        Core training Logic
        Weights must be updated or either exception should be raised for training halt
        '''
        pass

    def predict(self, new_data_point):
        new_x = self.preprocess(new_data_point)  # Making new preprocess

        if self.weights is None:
            raise Exception("Model is not trained | Terminating prediction")

        # Checking weights dimension
        if self.weights.shape[0] == self.d:
            raise Exception("Weight Shape is Not with convention")
        log.info(f'Prediction Shot : {self.weights.shape} with datapoint: {new_x.shape}')
        return new_x @ self.weights

        
