'''
Simple Visualization tool for all units, working as simple visualization plugin

Making layered visuals with explanation log and descriptive images

Approach
    Simple tools for loading functional values into graph
    Layered ploting
    
    Framing with simple api to handle graphing details
    Ploting the final plot

With object higherchy given with matplotlib 

ax being the axes which we can take into account for the plot and make multiple plots into one figure

Top down levels

figure
    |- Axes
        |-x1
        |-y1
        .
        .
Making multiple graph define subplots ranges
making one single graph define within one first axes
'''

import matplotlib.pyplot as plt
import numpy as np 

class VizionGraph:
    def __init__(self, title="vizion-plot"):
        '''
        With subplots having numerical way of columns and rows of plots which we might make, are the way around to make multiple plots into one sceen
        respective axes used for them
        ax.set_title("Specific plot title")
        fig.tightlayout() : Making ploting tight together
        '''
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.title = title

    def data_plot(self,x,y, label="Training Data", color="darkblue"):
        self.ax.scatter(x,y, color=color, alpha=0.6)
        return self

    def line_plot(self, weights, model):
        '''
        Using Existing Prediction function from model to plot the dataset

        BUG: Either the core working for building modle is broken
        Loading for prediction pipeline in Broken
        Issues
            + Model is not working | since prediction line is broken
            + Visualization part is also not working fine

        with first and last points predictions work somehow ?
        How really which is the prediction part ? what is really that prediction line which we talk abouut ?
        '''
        simple_data_points = model.features
        predictions = []
        # Taking first and last for the predictions
        first = simple_data_points[0]
        last = simple_data_points[len(simple_data_points) - 1]


        p1 = model.predict([first])
        p2 = model.predict([last])
        preds = [p1, p2]
        base = [first, last]
        self.ax.plot(base, preds)

    def render(self):
        plt.title(self.title)
        plt.show()

