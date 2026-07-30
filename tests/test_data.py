"""
Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""
from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare

from src.utility.CsvLoader import CSVHandle
import numpy as np 

'''
Input Domain, What should be really the vectors be like
Simple input with this list is breaking internals with shape missmatch

What is [[]] <-- Being inside the list ?
'''
#X = [1, 2, 3, 4, 5]
#Y = [1, 3, 2, 5, 4]

# Loading Dataset from csv


print("Executing Ordinary least squares Method")

X,Y = CSVHandle('/home/madhav/workspace/projects/yantragya/tests/test_data.csv', "y").load()
model = OrdinaryLeastSquare(X,Y)
w =model.train_weight()
print(f" After training weights  : {w}")

