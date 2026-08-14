# Gradient Descent Algorithm implementation
**goal : find weights in efficient way without computing with matrices**

## Targets
1. Making batch processing pipeline with dataset
2. Understanding the working of gradient descent algorithm

## Preface
An algorithm for iterativly moving towards steepest accent in the curve, defined by negetive gradient,

In our context target function : least squares algorithm
We iterativly go towards the minimum direction for given function,to find a point satisfactory with validation calculations

## Gradient Descent Derivation

Model prediction 
$$
y^h = wx_i
$$

Target Loss function
$$
L=\sum_i^n ( wx_i − y_i )^2
$$

Matrix based representation and differentiation

chain rule implemented here 🔖
> Since the target variable is inside the quadratic
> We first take the outside function derivative resulting 2
> we simply took derivative of internal function and wrote final product

$$
\begin{align}
    L = (w^TX - y)^2 \\
    & = 2(w^T - y).X
\end{align}
$$

Differentiated function
$$
f(w) = 2(w^T - y).X
$$

With Algebraic expension 👾
Note both are same just notation and context are different
$$
\sum_i^n \frac{1}{n} 2(wx_i-y_i)X
$$

**Gradient Descent Core formula** ♥️

$L(w)$ is the loss function computed with given weight, $\eta$ is the step size for the alogorithm to work with 

## Gradient CORE Algorithm🚀

$$
w_{new} = w_{old} - \eta \Delta L(w)
$$

Gradient Descent being a efficient way to compute best weights for given dataset for further prediction goals, it still holds some interesting thing worth knowing
* Batches and computation advantages
    We can also approach making further edit into algorithm during our practice with it, since we can also decide with which range we want to approach the given problem.
* Hyper parameters into the play
    Even being efficient we are ultimatly training weights thus we can also improve our core algorithm using constraints of ridge and lasso regression

**Implementation Note**
1. With one shot loading update sequence
    Since we can also approach to update weight with whole data iteration into one shot we would first implement this.
2. With batches based loading mechanism to try to test with big dataset loading for training the model
3. Implementing kernelaization | PCA implementation and constraints application into the core algorithm to make it robust and try to explore ways to integrate all these into one for simplicity.
