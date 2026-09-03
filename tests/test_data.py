"""
Testing whole working units
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#from src.vizion.transformation import *

from src.models.LinearRegression.GradientDescent import GradientDescent
from src.util.basic_utility import *

#df = pd.read_csv("/home/madhav/workspace/projects/yantragya/tests/needs_normalization.csv")

df = pd.read_csv('/home/madhav/workspace/projects/yantragya/data/cleaned_clean_data.csv')

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

y_train = np.array(df['y'].iloc[:cutoff]).reshape(-1,1)
y_test = np.array(df['y'].iloc[cutoff:]).reshape(-1,1)

# Yantragya Model
#model = OLS(X_train, y_train)
#model.train()

# Basic Requirements for Gradient Descent to work with :P
X_train, y_train = np.array(X_train), np.array(y_train)

y_train = y_train.reshape(-1,1)  # Needs to make a schema based shaper formating

model = GradientDescent(X_train, y_train)
w_initial = np.array([3,3]).reshape(-1,1)
w,err,ep = model.train(initiating_weight=w_initial)

# Gradient Descent Error Graphing
E = np.array(err)
ep = np.array(ep)

# Preparing X testing for final prediction
X_test = np.array(X_test)
X_test_raw = X_test  # taking this for visualization
X_test = X_test.reshape(-1,1)  # FIX: Made basic fix for this to run
sample = X_test.shape[0] # number of elems
ones_column = np.ones((sample, 1))
X_test = np.hstack([ones_column, X_test])

# hard coding X_test dimension
print(f"w with test : {w.shape} with test : {X_test.shape}")
predictions = X_test @ w

'''
# BACK TRANSFORMATION OF PREDICTION and target
predictions = (predictions * (training_max_y - training_min_y)) + training_min_y
y_test = (y_test * (training_max_y - training_min_y)) + training_min_y

'''

# Error computed
preds = np.array(predictions)
error = (preds - np.array(y_test)) ** 2
print(f"Final computed sse : {np.sum(error) / len(preds)}")

# Plot view
fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(X_train,y_train, color='blue')
ax[0].plot(X_test_raw, predictions, color='red')
ax[0].scatter(X_test_raw, y_test, color='orange')


ax[1].plot(ep, E, color='red')

plt.show()  # Error function graph

