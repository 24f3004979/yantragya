'''
Foundational Prototypic modules based functions
helping with non-scopic domain for the proeject

- Before using utility or touching the dataset
    First clean with bash script based pathway
'''
import pandas as pd 
import numpy as np

def normalize(data_column):
    '''
    data_column : Must be numerical
    Making min-max based normalization
    '''
    df = data_column
    norm_df = data_column.copy()
    min_val = df.min()
    max_val = df.max()

    norm_df = (df - min_val) / (max_val - min_val)
    return norm_df

    
