"""
Testing whole working units - Custom OLS vs Sklearn LinearRegression
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.models.LinearRegression.OLS import OLS
from src.util.basic_utility import *
# 1. Import scikit-learn's LinearRegression
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/madhav/workspace/projects/yantragya/cleaned_clean_data.csv')

# Spliting Dataset for testing | 80-20
cutoff = int(len(df) * 0.8)

X_train = df['X'].iloc[:cutoff]
X_test = df['X'].iloc[cutoff:]

y_train = df['y'].iloc[:cutoff]
y_test = df['y'].iloc[cutoff:]

# ==========================================
# 🛠️ RUNNING YOUR CUSTOM OLS MODEL
# ==========================================
custom_model = OLS(X_train, y_train)
custom_model.train()

custom_predictions = custom_model.predict(X_test)
custom_preds_arr = np.array(custom_predictions)
custom_sse = np.sum((custom_preds_arr - np.array(y_test)) ** 2)

print("==== CUSTOM OLS RESULTS ====")
print(f'Model Weights : {custom_model.weights}')
print(f"Final computed sse : {custom_sse}\n")


# ==========================================
# 🤖 RUNNING SKLEARN LINEAR REGRESSION
# ==========================================
# Sklearn requires X to be a 2D array/DataFrame shape (N, 1) instead of a 1D Series
X_train_2d = X_train.to_numpy().reshape(-1, 1)
X_test_2d = X_test.to_numpy().reshape(-1, 1)

sklearn_model = LinearRegression()
sklearn_model.fit(X_train_2d, y_train)

sklearn_predictions = sklearn_model.predict(X_test_2d)
sklearn_sse = np.sum((sklearn_predictions - np.array(y_test)) ** 2)

print("==== SKLEARN OLS RESULTS ====")
print(f'Intercept (Bias β0) : {sklearn_model.intercept_}')
print(f'Slope Coefficient (β1): {sklearn_model.coef_[0]}')
print(f"Final computed sse  : {sklearn_sse}\n")


# ==========================================
# 📊 PLOT BOTH TO COMPARE THE FIT
# ==========================================
plt.figure(figsize=(10, 6))

# Data points
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Train Data')
plt.scatter(X_test, y_test, color='orange', alpha=0.7, label='Test Data')

# Model prediction lines
plt.plot(X_test, custom_predictions, color='red', linewidth=2, label='Custom OLS Line')
plt.plot(X_test, sklearn_predictions, color='green', linestyle='--', linewidth=2, label='Sklearn Line')

plt.title('Custom OLS vs Sklearn Linear Regression', fontsize=14, fontweight='bold')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()
