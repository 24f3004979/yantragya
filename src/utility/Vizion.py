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

    def line_plot(self, weights, dat_matrix):
        '''
        Simple line plot with weights
        Testing to make predictions with iterating all points to generate prediction plot
        '''
        preds = []
        for _ in dat_matrix:
            d = _[1]
            slope = weights[0]
            p = slope * d + weights[1]  # Line equation output
            print(f"Prediction : {p}")
            preds.append(p)
        print(f"Predictions : {preds}")
        self.ax.plot(preds) # Made plot for predictions


    def render(self):
        plt.title(self.title)
        plt.show()

