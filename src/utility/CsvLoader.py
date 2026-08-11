'''
CSV file Loading Module
very simple loader utility | Does not does good things to data to be that usefull

Needs scope clarity about what to implement with this module
Currently not used for the core workflow
Needs refactor for being assured about what it does and where it breaks with grace

Needed Featuring
1. Optimization with loading big dataset
    Dedicated pipeline to feature big  dataset loading without memory crash

    requires initial inspection module with defined loading pipeline with cleaning utility fucntion

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
        df.dropna()
        # target check
        if self.target not in df.columns:
            raise ValueError("Target Not found in Dataset | Terminating process")
        print(f"Data set loaded into pandas")
        
        # Core essentials :)
        X = df["X"] # | df.drop(columns=[self.target]) | Making problem of picking leading as the point
        y = df[self.target]
        y = y.fillna(0)
        
        # Numpy conversion
        X = X.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)



        if X.shape[0] != y.shape[0]:
            raise ValueError("Shape Missmatch | Terminating Process")

        print(f"Data Points : {len(X)} with lable : {len(y)} | one elem : {X[0]} with y element : {y[0]}")
        # FIX : Pandas Grabing function was making problem with loading dataset for training

        return (X,y)  # Final Feature and target values

