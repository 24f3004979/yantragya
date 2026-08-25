'''
Gradient Descent with Base ML
Batch processing for finding optimal weight
'''
from src.element.model import BaseML


class GradientDescent(BaseML):
    '''
    parameter format
    epotch, batch_size, eta

    '''
    def __init__(self):
        super().__init__()

    def train(self):
        initiating_weight = self.preprocess(self.hyper_para['initial_weight'], intercept_required=False)

        X = self.data
        y = self.target

        n = self.X.shape[0] # total data point
        weight = initiating_weight.reshape(-1,1)  # weight vectorized
        self.weight = weight

        er = []
        ep = []
        iteration = 0

        for epotch in range(epotches):

            indices = np.arange(n)
            np.random.shuffle(indices)
            X_shuf = X[indices]
            y_shuf = y[indices]


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
                err = (np.sum((pred - y.reshape(-1,1))**2) / len(X))

                ep.append(iteration)
                er.append(err)
                print(f"Adding information for error visualization : {err} with {iteration} round")
                iteration += 1

            print(f'Epotch : {epotch} with sse : {err}')
        return weight, er, ep

