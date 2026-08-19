"""Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""
from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare
from src.LinearRegression.GradientDescent import GradientDescent

from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare

from src.utility.CsvLoader import CSVHandle
import numpy as np 
# vizion module is currently broken and moved for archival :) | Pririty shitft 
#from src.utility.Vizion import VizionGraph

import matplotlib.pyplot as plt

# Running Debugger 
X = CSVHandle('/home/madhav/workspace/projects/yantragya/tests/test_data.csv', "y").load()

training_data = {
        "X" : X,
        "y" : Y
        } # json :)

seed_weight = [3,3]
plt.scatter(X, Y)

model = OrdinaryLeastSquare(X, Y)
weights = model.train_weight()

print(f"Predicted Weights  : {weights}")

# Making some basic edit with being into insert mode Geting into escape with the 

X_flatten = X.flatten()
first = X_flatten[1]
last = X_flatten[-1]
print(f'first : {first, last}')
pred1 = (first * weights[1]) + weights[0]
pred2 = (last * weights[1]) + weights[0]
predictions = [[first, last], [pred1, pred2]]

print(f"Predictions : {predictions}")
cords = [first, last]
preds = [pred1, pred2]
plt.plot(cords, preds, color='red')
plt.show()

#gradient_model = GradientDescent(training_data, seed_weight)
#p = gradient_model.y_cap(seed_weight)

#gradient_model.train(10, 0.2)
