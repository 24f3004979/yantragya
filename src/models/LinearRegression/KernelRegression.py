'''
Kernel Regression Model
Loading with kernalization module set
'''
from src.element.model import BaseML
from src.models.LinearRegression.GradientDescent import GradientDescent

class KernelRegression(BaseML, GradientDescent):
    def __init__(self):




