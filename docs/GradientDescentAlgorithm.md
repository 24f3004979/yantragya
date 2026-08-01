<link rel="preconnect" href="https://googleapis.com">
<link rel="preconnect" href="https://gstatic.com" crossorigin>
<link href="https://googleapis.com/css2?family=JetBrains+Mono&display=swap" rel="stylesheet">

<style>
  body, p, h1, h2, h3, h4, h5, h6, li, code {
    font-family: 'JetBrains Mono', monospace !important
  }
</style>

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

model prediction ~
$$
y^h = wx_i
$$

Loss for given model ~
$$
L=\sum_i^n​ \frac{1}{n}(wx_i​−y_i​)^2.
$$

Differentiated version

$$
\frac{d}{dw}(f(w))^2 = 2f(w)f^`(w)
$$
with chain rule with nested such function first we take derivate of outer function and multiply with inner function and then take derivative of inner function and multiply it with other function

$$
\sum_i^n \frac{1}{n} 2(wx_i-y_i)x_i
$$

**Gradient Descent Core formula** ♥️

$L(w)$ is the loss function computed with given weight, $\eta$ is the step size for the alogorithm to work with 

🚀 || Gradient CORE || 🚀
$$
w_{new} = w_{old} - \eta \Delta L(w)
$$
