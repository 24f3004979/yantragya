import numpy as np

class GradientDescent:
    def __init__(self, X, y, d=1):

        # BUG : Ensure to set d if multi dim
        X = X.reshape(-1,d) 
        
        sample = X.shape[0] # number of elems
        ones_column = np.ones((sample, 1))
        self.X = np.hstack([ones_column, X])
        self.y = y.reshape(-1,1)

    
    def train(self, initiating_weight, epotches=10, batch_size=100, eta=0.0001):
        '''
        Training Prediction with batch based approach
        Now geting nan in between
        '''
               
        n = self.X.shape[0] # total data point
        weight = initiating_weight.reshape(-1,1)  # weight vectorized

        er = []
        ep = []
        iteration = 0

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
                pred = (x).dot(weight)
                print(pred)
                print(f"Predictions being zero :)")
                err = (pred - y)
                print(f"Intermediate error for intermediate prediction : {err}")

                gradient = (2.0 / batch) * np.dot(x.T, err)

                # regularization required here :)
                print(f'gradient intermediate computation :{(eta * gradient)}')
                weight = weight - (eta * gradient)
                
                pred = weight.T * self.X
                err = (np.sum((pred - self.y.reshape(-1,1))**2) / len(self.X))

                ep.append(iteration)
                er.append(err)
                print(f"Adding information for error visualization : {err} with {iteration} round")
                iteration += 1
        
            print(f'Epotch : {epotch} with sse : {err}')
        return weight, er, ep 
