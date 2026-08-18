# PCA
Way to simplify dataset, Domain of representational learning
Dataset -> Covariance Matrix --> Its most critical directions

## Dive into core 
Part of representational learning :
representing original dataset with help of principle components which are derived as the directions with most information.

Thus we can reduce the dimension to represent the dataset
component : eigen vectors of covariance matrix
eigen vectors are the special vector direction which could be used to represent the space where given matrix transforms its input vector

Maping the original dataset into another space with the help of such special vectors [ eigen vectors ]

**Co-variance matrix**
Matrix which represents the pair wise change of its elements. central diagonal is for the self elemental status, off diagonal elements represents the relation with other elements.

Numerical extracts
with each components co-variance representing there co-variations, how much they would change with each other, with negetive they would have different directions, with positive they are prototional,and with zero they dont have any such good relations which we can say about.

### Critical Flow for PCA 
1. First take average for each numerical listing
2. center the dataset
3. Compute dot products | Covariance matrix
    off diagonal elements : relation with pair wise elements
    diagonal : variance of each given variables
4. compute the eigen vector and eigen values for final computation
5. Use the eigen vector matrix to project the original vector dataset into k-dimensional sub-space
6. Train model with new space with fewer dimensional load and clear pathway

PCA driven pipeline
dataset > center them > covariance matrix > eigen vectors 

rank eigen vectors and pick k-dimensional matrix
project data point into new dimension > train the new pca driven model

For new points
must use original mean from training dataset > dependency link
project into pca space -> Make predictions

