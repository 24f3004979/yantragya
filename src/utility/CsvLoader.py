'''
CSV file Loading Module

Early termination if Dataset is not ready for operation
'''
import pandas as pd 
import numpy as np 

class CSVHandle:
    '''
    Simple utility function related to data loading
    columns validations
    missing termination
    integrity checks
    '''
    def __init__(self, file_path, target):
        self.file_path = file_path
        self.target = target

    def load(self):
        '''
        Core Loading function for loading dataset for the sub-functions working
        validate non-missing data points
        ir-regular shapes
        target-missing
        messy dataset

        early terminations with such conditions
        '''
        df = pd.read_csv(self.file_path)
        # target check
        if self.target not in df.columns:
            raise ValueError("Target Not found in Dataset | Terminating process")
        print(f"Data set loaded into pandas")
        
        # Core essentials :)
        X = df.drop(columns=[self.target])
        y = df[self.target]
        

        # Numpy conversion
        X = X.to_numpy(dtype=float)
        X = y.to_numpy(dtype=float)

        print(f"Numpy conversion completed")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Shape Missmatch | Terminating Process")
        print(f"Final Data : {X} with {y}")
        return (X,y)  # Final Feature and target values

