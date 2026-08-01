"""Making simple data load for Linear regression model

initiate simple noisy dataset for the linear regression testing
- test Linear regreession equation
- Build simple visualization tunel with the unit
"""
from src.LinearRegression.OrdinaryLeastSquare import OrdinaryLeastSquare

from src.utility.CsvLoader import CSVHandle
from src.utility.Vizion import VizionGraph
import numpy as np 



X,Y = CSVHandle('/home/madhav/workspace/projects/yantragya/tests/test_data.csv', "y").load()

graph = VizionGraph()
graph.data_plot(X,Y)

model = OrdinaryLeastSquare(X,Y)


w =model.train_weight()  # training weight
graph.line_plot(w,model)
graph.render()
