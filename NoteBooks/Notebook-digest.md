# Note Book Extract
Extract from notebooks, insights and learning form all notebooks and documenting them into format usefull for further experiments in notebook

## Modeling Distribution and taking samples

Random Number Generator Object
`rng = np.random_default_rng(seed=101)`
Gives us a simple function which to call for generating random number lst

With choice with p thing we can model out bernouli like distribution being made with simple aproach.

`X = rng.choice([0,1], size=1000, p=[0.2, 0.8])`
With testing through finding best estimate for p we assure this

