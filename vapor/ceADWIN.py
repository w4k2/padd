import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ceADWIN(BaseEstimator, ClassifierMixin):
    def __init__(self, delta = 0.0001):
        self.delta = delta
        self.drift = []

    def feed(self, X, cert):

        if not hasattr(self, "mu_W"):
            self.W = np.copy(X)
            self.c = np.copy(cert)
            self.mu_W = []
            self.sizes = []
            self.drift.append(0)
        else:
            self.W = np.append(self.W, X, axis=0)
            self.c = np.append(self.c, cert, axis=0)
            values = np.array(self.c)
            var = np.var(values)
            delta_p = self.delta/self.W.shape[0]

            step = int(np.sqrt(self.W.shape[0]))

            self.isdrift = False
            for i in range(1, self.W.shape[0], step):
                m = 1/((1/self.W[:i].shape[0]) + (1/self.W[i:].shape[0]))
                uw0, uw1 = np.mean(values[:i]), np.mean(values[i:])
                cut = np.sqrt((2/m) * var * np.log(2/delta_p)) + (2/(3*m)) * np.log(2/delta_p)

                print( np.abs(uw0 - uw1))
                if np.abs(uw0 - uw1) >= cut:
                    self.W = self.W[i:]
                    self.c = self.c[i:]
                    self.drift.append(2)
                    self.isdrift = True

                    break
            if self.isdrift == False:
                self.drift.append(0)

        self.mu_W.append(np.mean(self.W))
        self.sizes.append(self.W.shape[0])

        return self