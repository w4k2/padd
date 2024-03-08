import numpy as np
from sklearn.base import clone
from scipy.stats import ttest_ind

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    return x * (x > 0)

class RandomSight():
    def __init__(self, n_features, n_outputs, neck_width=512):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.neck_width = neck_width
        
        self.stack = [
            np.random.normal(0,.1,
                             (self.n_features+1,
                              self.neck_width)),
            np.random.normal(0,.1,
                             (self.neck_width,
                              self.neck_width)),
            np.random.normal(0,.1,
                             (self.neck_width,
                              self.n_outputs)),
        ]
    
    def predict_proba(self, X):
        # Copy input
        val = np.concatenate((np.copy(X), 
                              np.ones((X.shape[0], 1))), axis=1)
        # Propagate through layers
        for layer_id, layer in enumerate(self.stack):
            val = relu(val @ layer)
        
        # Calculate softmax
        predict_proba = val
        predict_proba = np.array([softmax(v) for v in val])
        
        return predict_proba

class CDET:
    def __init__(self, base_mlp, n_epochs, alpha, mps_th, balanced=False, psk=True):
        self.base_mlp = clone(base_mlp)
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.balanced = balanced
        self.psk = psk
        self.mps_th = mps_th
    
        self.is_drift = None
        
    def process(self, X, y):
        
        #setup on 1st chunk
        if self.is_drift is None:
            self.prevs = []
            self.is_drift = False

            self._train(X, y)

        else:
            ps = np.max(self.base_mlp.predict_proba(X), axis=1)    
            if len(self.prevs)>0:
                
                _prevs = np.array(self.prevs).flatten()
                
                # test hipotez
                if self.balanced:
                    _prevs = np.random.choice(_prevs, len(ps), replace=False)
                
                _, p_val = ttest_ind(ps, _prevs)
                # print(p_val)

                if p_val<self.alpha: 
                    print(p_val)
                    self.is_drift = True
                    self.prevs = []
                    
                else:
                    self.is_drift = False
                    self.prevs.append(ps)
                    
            else:
                self.is_drift = False
                self.prevs.append(ps)

        if self.is_drift:
            self._train(X, y)
            
        
    def _train(self, X, y):
        if self.psk:
            counter=0
            while(1):
                counter+=1
                self.base_mlp.partial_fit(X, y, np.arange(2))
                mps = np.mean(np.max(self.base_mlp.predict_proba(X), axis=1))
                if mps>=self.mps_th:
                    break
                if counter>100000: #zapobiegawczo
                    break
        else:
            [self.base_mlp.partial_fit(X, y, np.unique(y)) for e in range(self.n_epochs)]
            