"""
Testing whole working units
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#from src.vizion.transformation import *

from src.models.LinearRegression.GradientDescent import GradientDescent
from src.util.basic_utility import *

from sklearn.linear_model import LinearRegression

#df = pd.read_csv("/home/madhav/workspace/projects/yantragya/tests/needs_normalization.csv")

df = pd.read_csv('/home/madhav/workspace/projects/yantragya/cleaned_clean_data.csv')

# Normalization | Transformation
'''
#Making Transformation with normalization would also break graphing
training_min_y = df['y'].min()
training_max_y = df['y'].max()
df['X'] = normalize(df['X'])
df['y'] = normalize(df['y'])
'''


# Spliting Dataset for testing | 80-20
cutoff = int(len(df) * 0.8)

X_train = df['X'].iloc[:cutoff]
X_test = df['X'].iloc[cutoff:]

y_train = df['y'].iloc[:cutoff]
y_test = df['y'].iloc[cutoff:]

# Yantragya Model
#model = OLS(X_train, y_train)
#model.train()

# Basic Requirements for Gradient Descent to work with :P
X_train, y_train = np.array(X_train), np.array(y_train)

y_train = y_train.reshape(-1,1)  # Needs to make a schema based shaper formating

model = GradientDescent(X_train, y_train)
w = np.array([3,3])
w = model.train(initiating_weight=w)

# STACKING ONE INTO X FOR INTERCEPT
sample = X_test.shape[0] # number of elems
ones_column = np.ones((sample, 1))
self.X = np.hstack([ones_column, X_test])

# hard coding X_test dimension
X_test = X_test.reshape(-1, 1)
predictions = X_test @ w

'''
# BACK TRANSFORMATION OF PREDICTION and target
predictions = (predictions * (training_max_y - training_min_y)) + training_min_y
y_test = (y_test * (training_max_y - training_min_y)) + training_min_y

'''

# Error computed
preds = np.array(predictions)
error = (preds - np.array(y_test)) ** 2
print(f"Final computed sse : {np.sum(error)}")

# Plot view
plt.scatter(X_train,y_train, color='blue')
plt.plot(X_test, predictions, color='red')
plt.scatter(X_test, y_test, color='orange')

plt.show()
