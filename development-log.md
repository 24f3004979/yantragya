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
Visualization tool is implemented with being a wrapper for the matplotlib

---- training error with some wrong computation or core alogorithm being wrong---
Core algorithm giving weights are inconsistent with the data set , making some concern about the total training part

Q : How does the prediction with taking first and last point works ? 
    How really does the prediction works with the modle ? 
    How to validate the model with the predictions ?

--- Needs debug for the data flow
++ Visualization pipeline is not working as intended

---

# Gradient Descent Implementation
Made a whole day debuging the flow with gradient descent algorithm, geting nan at results with using computing of umpy to compute the final weights,

Inspected the whole pipeline about where things are geting wrong, laid out with clarity about the shapes and flow of numbers, then things started working

**Numpy extractions are also reqired : currently i have made flatten() function based extraction from its array capsule like [[]] after matrix multipication, need to learn to make good flow with numpy

Then final fix was to either fill y with 0 in place of nan , or remove them , to save the shape chaos i simply replaced the y with 0 at all nan places, and thing started working

Now i would try to verify its working with real library working about the same problem 

---
# Refactor : [5| august | 2026]
Targets to refactor into code base
1. Making structured documentation - rewriting into readable, and structured way [ Made foundational documentation corrections to all components and now needs written plan and documentation strategy to approach other components design]
2. Dedicated componenets requried in all core componenets | Too much responsibility at once loaded [
initial planing is into action for making responsibility allocations for all modules and unit
]
3. Loging inspection requried into the  main workflow with testing units | since shapes and flow makes non expected crashes [ Simple loger would be soon loaded into root folder]
4. Breakdown of core componenets and planed mvp componenets requrired for the working 

------
kernel Initialization command required usefull thing
```
python -m ipykernel install --user --name=kernel --display-name="kernel-core"
```
Dated : 11 august 2026
Defined plan with documentation required for the master project flow 
Goals | Scope | Designed components | research documentations

### Notes from exploration
Currently we require reading materials study about the topics, implementation clarity is missing for all designs currently made, thus we need good plan with refinements and learning from current implementation with structure about controlled experimentation, to achive goals


