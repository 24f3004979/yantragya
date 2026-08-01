"""Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""
from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare

from src.utility.CsvLoader import CSVHandle
import numpy as np 
from src.utility.Vizion import VizionGraph

import matplotlib.pyplot as plt




X,Y = CSVHandle('/home/madhav/workspace/projects/yantragya/tests/test_data.csv', "y").load()
x_new, y_new = CSVHandle("/home/madhav/workspace/projects/yantragya/linear_regress.csv", 'y').load()

plt.scatter(x_new, y_new)

# plt.scatter(X,Y)
model = OrdinaryLeastSquare(x_new,y_new)
model.train_weight()
w = model.train_weight()  # training weight

# Simple prediction flow
preds = []
for i in model.features:
    p = model.predict([i])
    preds.append(p)

print(f"With Weigtht : {model.w}")
print(f"Prediction Status : {preds}")
plt.plot(model.features, preds, label='Model Predictions', color='red')
plt.show()

