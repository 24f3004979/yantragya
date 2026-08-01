# OLS
A way to compute best weight for given dataset with matrix way
With limmitations of least square to find best weights, if data matrix is non-invertible, OLS comes with concept of pseudo inverse which elliminates the need to compute explicit inverse.

## A bit of understanding for the OLS model
We project target vector to feature space, to get the best representation of target vector from the feature space, Projection is linear combination of feature space spanning vectors.

Video Demonstration for given concept : []  # upcomming :)


### Mathematical Limmitations for OLS method
We cant run this algorithm for big data : Matrix computations are costly
Our working prototype also terminates to compute weights somehow :) | BUG reported

+ Extream Overfitting
    If with higher dimension it works then it would overfit with dataset

    **High Dimensional Linear modeling is extream fit** : Hyper planes
    with increased number of features then data points, Linear model becomes free with dimensions and could get into situation of representing all data points exactly | Overfiting

#### Unformal mathematical proof of overfiting with high dimensional dataset

with more features, then number of points, the feature space would span the whole n dimensional space, thus we would get exact fit with the feature vectors for the target vector with no projection this model wont work well with new dataset

+  No uniqueness
    Inverse method picks best minimizing weights with not structure

Fixes Possibility
+ Running PCA beforehand for compressed representation for the high dimentional dataset
+ Lasso + ridge regression methods for constraints for over fit



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


### KNOWN LIMMITATIONS AND PROBLEMS - CODE BASE SPECIFIC
+ With Loaded dataset of big size : weight comes as Nan
+ Non-predictive workflow about module : logs required for the working
+ training crashes or stops without warning and traces :)

