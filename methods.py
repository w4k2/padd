import numpy as np
from sklearn.base import clone
from scipy.stats import ttest_ind


class CDET:
    def __init__(self, base_mlp, n_epochs, alpha):
        self.base_mlp = clone(base_mlp)
        self.n_epochs = n_epochs
        self.alpha = alpha
    
        self.is_drift = None
        
    def process(self, X, y):
        
        if self.is_drift is None:
            #setup on 1st chunk
            # print('TRAINING 1st chunk')
            self.prevs = []
            
            self._train(X, y)
            self.is_drift = False
            
            ps = np.max(self.base_mlp.predict_proba(X), axis=1)    
            self.prevs.append(ps)

        else:
            ps = np.max(self.base_mlp.predict_proba(X), axis=1)    
            # test hipotez
            stat, p_val = ttest_ind(ps, np.array(self.prevs).flatten())     
            # print(stat,p_val)

            if p_val<self.alpha:                    
                self.is_drift = True
                self.prevs = []
            else:
                self.is_drift = False
                self.prevs.append(ps)

        if self.is_drift:
            # print('TRAINING drf')
            self._train(X, y)
            
        
    def _train(self, X, y):
        [self.base_mlp.partial_fit(X, y, np.unique(y)) for e in range(self.n_epochs)]
            
class CDET_PSK:
    def __init__(self, base_mlp, mps_th, alpha):
        self.base_mlp = clone(base_mlp)
        self.mps_th = mps_th
        self.alpha = alpha
    
        self.is_drift = None
        
    def process(self, X, y):
        
        if self.is_drift is None:
            #setup on 1st chunk
            # print('TRAINING 1st chunk')
            self.prevs = []
            
            self._train(X, y)
            self.is_drift = False
            
            ps = np.max(self.base_mlp.predict_proba(X), axis=1)    
            self.prevs.append(ps)

        else:
            ps = np.max(self.base_mlp.predict_proba(X), axis=1)    
            # test hipotez
            stat, p_val = ttest_ind(ps, np.array(self.prevs).flatten())     
            # print(stat,p_val)

            if p_val<self.alpha:                    
                self.is_drift = True
                self.prevs = []
            else:
                self.is_drift = False
                self.prevs.append(ps)

        if self.is_drift:
            # print('TRAINING drf')
            self._train(X, y)
            
        
    def _train(self, X, y):
        counter=0
        while(1):
            counter+=1
            self.base_mlp.partial_fit(X, y, np.arange(2))
            mps = np.mean(np.max(self.base_mlp.predict_proba(X), axis=1))
            if mps>=self.mps_th:
                break
            if counter>100000: #zapobiegawczo
                break