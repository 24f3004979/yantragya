from src.models.LinearRegression.GradientRegression import GradientDescent
import pandas as pd
import matplotlib.pyplot as plt

parameters = {
        "epotches" : 10,
        "batch_size" : 100,
        "eta" : 0.0001,
        "initiating_weight" : [3,3]
        }

df = pd.read_csv('/home/madhav/workspace/projects/yantragya/cleaned_clean_data.csv')


Gradient_Unit = GradientDescent()

X = df['X']
y = df['y']

Gradient_Unit.load(X, y, parameters)

err_list = Gradient_Unit.train() # Inheritense final testing

iterations = range(0,parameters['epotches'])

plt.scatter(X, y, color='blue', alpha=0.2) # original training dataset

t1 = X.iat[0]
t2 = X.iat[-1]

p1 = Gradient_Unit.predict(t1)
print(f"Predictin point : {p1}")

print(f"Final trained weight : {Gradient_Unit.weights}")




