"""
Testing whole working units
"""
import pandas as pd

from src.vizion.transformation import *

df = pd.read_csv("/home/madhav/workspace/projects/yantragya/tests/needs_normalization.csv")

targets = ['Income', 'Age']
t = TransformationHandle(df, targets)

nrmals = t.normalize()
nrmals.plot()
print(nrmals.head())
