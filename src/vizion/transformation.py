'''
General transformation tools for dataset
making simple transformation for the dataset for easy visualization and comprehension

+ Outlier removal | IQR based
+ Normalization | simple numerical changes
+ If dataset is into high dimension | correlation conditioned
'''
import pandas as pd
import numpy as np

class TransformationHandle:
    def __init__(self, dataframe: pd.DataFrame, target_columns :list):
        '''
        dataframe : pandas data frame
        target_columns : [targets]
        '''
        self.df = dataframe
        self.X = self.df[target_columns]

    def normalize(self):
        '''
        Making normalization for numerical inputs
        returning simple visualization friendly numbers

        Normalization works for making minmax normalization
        '''
        num_col = self.X.select_dtypes(include=[np.number]).columns.tolist()
        df_norm = self.df.copy()
        for col in num_col:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()

            if max_val == min_val:
                df[col] = 0.0
            else:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

        return df_norm


