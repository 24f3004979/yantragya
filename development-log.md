# Dev Log book
Linear Regression
    Simple for small dataset due to computing needs
    Computing Least square is great way to make Linear Model for small dataset

    With increasing datasize we use Gradient Decent algorithm for geting into the ideal linear model.

    With gradient decent way we nudege the line into the data frame with nudge, with multiple loops with the calculation, thus epoch is required for finding the saturation for the main data-frame.

Least square way : Awesome for small dataset which can fit into the memory frame at once for processing , the same reason its not used for big dataset to compute the best fit.

-------------------

## BUG: Shape Missmatch issue with Linear Regression OLS Module
Initiation Isssues with simple convertion into numpy array for further computation
- If we Initiate the Feature array with [] wrap, then further shape missmatch happens
- Initiation of onces is not stacking with the feature matrix
- Further computation is terminated with shape missmatch

### Root problems
No iteration of logic, initiation was breaking due to self carelessness, of converting the feature then loading raw list again :P 
Making object oriented miss-conception with non-defined values of y lying into training function
