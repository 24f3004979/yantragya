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
E = y - X \hat\{\beta}
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
