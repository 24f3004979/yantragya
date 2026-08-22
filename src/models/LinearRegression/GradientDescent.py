import numpy as np

class GradientDescent:
    def __init__(self, X, y, d=1):

        # BUG : Ensure to set d if multi dim
        X = X.reshape(-1,d) 
        
        sample = X.shape[0] # number of elems
        ones_column = np.ones((sample, 1))
        self.X = np.hstack([ones_column, X])
        self.y = y

        self.weight = None
    
    def train(self, initiating_weight, epotches=100, batch_size=100, eta=0.01):
        '''
        Training Prediction with batch based approach
        Now geting nan in between
        '''
        
        n = self.X.shape[0] # total data point
        self.weight = initiating_weight

        self.weight = self.weight.reshape(-1,1)

        for epotch in range(epotches):

            indices = np.arange(n)
            np.random.shuffle(indices)
            X_shuf = self.X[indices]
            y_shuf = self.y[indices]
            

            for i in range(0, n, batch_size):
                x = X_shuf[i: i+batch_size]
                y = y_shuf[i: i+batch_size]
                
                print(f"Current batch size: {x.shape[0]}")
                batch = x.shape[0]
                pred = x.dot(self.weight)
                err = pred - y

                gradient = (2.0 / batch) * np.dot(x.T, err)

                # regularization required here :)

                self.weight = self.weight - (eta * gradient)
                
                print(f'computed weight : {self.weight} batch {i}')

        pred = self.weight.T * self.X
        err = (np.sum((pred - self.y)**2) / len(self.X))
        
        print(f'Epotch : {epotch} with sse : {err}')
        return self.weight




        
