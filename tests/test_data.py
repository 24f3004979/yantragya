"""Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""
#from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare
from src.LinearRegression.GradientDescent import GradientDescent

from src.utility.CsvLoader import CSVHandle
import numpy as np 
# vizion module is currently broken and moved for archival :) | Pririty shitft 
#from src.utility.Vizion import VizionGraph

import matplotlib.pyplot as plt




X,Y = CSVHandle('/home/madhav/workspace/projects/yantragya/linear_regress.csv', "y").load()

seed_weight = [3,3]
gradient_model = GradientDescent(X, seed_weight)
p = gradient_model.y_cap(seed_weight)

for _ in p:
    print(f"Prediction Computed number : {_}")

