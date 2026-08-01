# OLS
A way to compute best weight for given dataset with matrix way
With limmitations of least square to find best weights, if data matrix is non-invertible, OLS comes with concept of pseudo inverse which elliminates the need to compute explicit inverse.

## A bit of understanding for the OLS model
We project target vector to feature space, to get the best representation of target vector from the feature space, Projection is linear combination of feature space spanning vectors.

Video Demonstration for given concept : []  # upcomming :)

## Project Implementation details about OLS

[updated : 1/Aug/2026]
> Working discription about current workflow
Breif workflow ~
```python
    X,y = CSV_Loader("Your dataset")
    model = OLS(X,y)
    model.train_weight() # Training weights

    model.predict([data_point])
```
### Internal Deatails
+ CSV_Loader : made for basic preprocessing and checks the data file for early termination or loads into columns for down-streams
+ model : Using simple numpy computations computes the weights and initiates predictions


### KNOWN LIMMITATIONS AND PROBLEMS
+ With Loaded dataset of big size : weight comes as Nan
+ Non-predictive workflow about module : logs required for the working
+ training crashes or stops without warning and traces :)

