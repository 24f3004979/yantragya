"""
Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""
from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare
X = [1, 2, 3, 4, 5]
Y = [1, 3, 2, 5, 4]

print("Executing Ordinary least squares Method")

model = OrdinaryLeastSquare(X,Y)


