import numpy as np
from sklearn.base import clone
from scipy.stats import ttest_ind


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
            