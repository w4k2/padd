import numpy as np
from scipy.stats import ttest_ind
from sklearn.base import clone

class certaintyDD:
    def __init__(self, base_clf, alpha=0.05, epochs=1):
        self.base_clf = clone(base_clf)
        self.alpha = alpha
        self.epochs = epochs

        self.is_drift = False
        
        self.prevs = []
    
    def detect(self, X):
        
        ps = np.max(self.base_clf.predict_proba(X), axis=1)    
        if len(self.prevs)>0:
            # test hipotez
            stat, p_val = ttest_ind(ps, np.array(self.prevs).flatten())     
            if p_val<self.alpha:                    
                self.is_drift = True
                self.prevs = []
            else:
                self.is_drift = False
                self.prevs.append(ps)
        else:
            self.prevs.append(ps)
        
    def partial_fit(self, X, y, classes):
        try: # 1st chunks
            self.detect(X)
        except:
            [self.base_clf.partial_fit(X, y, classes) for i in range(self.epochs)]
            return self
        
        if self.is_drift:
            # print('FITTING')
            [self.base_clf.partial_fit(X, y, classes) for i in range(self.epochs)]
        return self
    
    def predict(self, X):
        return self.base_clf.predict(X)
    
def get_real_drifts(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts
