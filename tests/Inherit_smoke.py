from src.models.LinearRegression.GradientRegression import GradientDescent
import pandas as pd

parameters = {
        "epotches" : 10,
        "batch_size" : 100,
        "eta" : 0.0001,
        "initial_weight" : [3,3]
        }

df = pd.read_csv('/home/madhav/workspace/projects/yantragya/cleaned_clean_data.csv')


Gradient_Unit = GradientDescent()

X = df['X']
y = df['y']

Gradient_Unit.load(X, y, parameters)

Gradient_Unit.train() # Final testing for OOPs ML




