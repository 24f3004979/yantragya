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
        self.target = target

## Shape Issue resolved
numpy shapes treats, list of number as one dimensional vector, with shape being (n,) thus it didnt worked with the rest of the module 
Solution : reshape function with (-1,1) being the value, which makes the given vector converted into one single vecotor with one column, -1 says to compute the row required for the given column, reshape, 

learning : we should be mind full about shapes of the data points moving within the defined space, weather it goes wrong or into right space

---- Implementing Simple visualization tool for modules ---
