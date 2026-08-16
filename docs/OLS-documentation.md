# OLS
A way to compute best weight for given dataset with matrix way
With limmitations of least square to find best weights, if data matrix is non-invertible, OLS comes with concept of pseudo inverse which elliminates the need to compute explicit inverse.

## A bit of understanding for the OLS model
We project target vector to feature space, to get the best representation of target vector from the feature space, Projection is linear combination of feature space spanning vectors.

[Video Demonstration](https://youtu.be/iOsGqscrVGI?si=UOi4bBNGBalVexcz)

# FIT

Finding Best weight for making best predictions

## Genre of fit

**Mathematical Models**

1. simple models fit equations | Linear Models
2. Optimization Algorithm
   ~ non-linear models weight
   ~ Gradient Decent algorithm for convergence
3. Probabilistic Algorithms
   ~ Finding Best distribution to fit into dataset
   ~ Bayesian Learning and normal distribution fitting.

## Linear Regression Core

**Finding best line fit for given dataset, calculus route we minimize the sum of squared errors for obtaining weights via differentiating with respect to m and c, it gives complexity for scale up with multi features and multi label.**

_Linear Algebra way_ Finding approximation for error vector and compute the required weights

$$
X {\beta} = y
$$

$\beta$ is the slope and intercept for the given data weights

Error Vector Minimization

$$
E = y - X \hat{\beta}
$$

Reducing Error Vector requires E vector to be orthogonal to the input vector

$$
X^Ty . E = 0
$$

Finding for given orthogonality final Equation for Linear Regression

$$
(X^TX^-1)X^Ty = \hat{\beta}
$$

### Some required ponder

1. Why we use the squared error mean for computing the error
   Best way to punish the errors along with maintain consistent integers
2. Why do we need to add one extra column filled with one into the main X vector
   To give the final fit freedom of finding intercept since if we wont include it the final fit would be forced to pass from the origin of the plot, thus adding intercept makes it easy for the final fit and predictions.
3. Orthogonal Vectors helps minimize
   > Geometrical explanation is required thus i would visualize it to understand how does actually orthogonal vector makes the difference.

    Orthogonal vectors are required because, they show the minimum distance from the target point to the given plane, thus we use the orthogonal projections.




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

