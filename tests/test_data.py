"""
Testing whole working units
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#from src.vizion.transformation import *
from src.models.LinearRegression.OLS import OLS

#df = pd.read_csv("/home/madhav/workspace/projects/yantragya/tests/needs_normalization.csv")

df = pd.read_csv('/home/madhav/workspace/projects/yantragya/ols_single_feature.csv')


# Spliting Dataset for testing | 80-20
cutoff = int(len(df) * 0.8)

X_train = df['X'].iloc[:cutoff]
X_test = df['X'].iloc[cutoff:]

y_train = df['y'].iloc[:cutoff]
y_test = df['y'].iloc[cutoff:]


model = OLS(X_train, y_train)
model.train()

predictions = model.predict(X_test)
print(f"Predictions {predictions}")

# Error computed
preds = np.array(predictions)
error = (preds - np.array(y_test)) ** 2
print(f"Final computed sse : {np.sum(error)}")

# Plot view
plt.scatter(X_train,y_train, color='blue')
plt.plot(X_test, predictions, color='red')
plt.scatter(X_test, y_test, color='orange')

plt.show()
