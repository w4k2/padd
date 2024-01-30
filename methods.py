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
    
    
class certaintyEns:
    def __init__(self, base_clf, n_clfs, alpha=0.05, epochs=1, soft=False):
        self.n_clfs = n_clfs
        self.alpha = alpha
        self.epochs = epochs
        self.soft = soft
        
        self.ensemble = [clone(base_clf) for n in range(self.n_clfs)]

        self.is_drift = False
        self.fitted = False
        self.prevs = [[] for n in range(self.n_clfs)]
    
    def detect(self, X):
        
        if self.soft==False:

            _is_drift = []
            for clf_id, clf in enumerate(self.ensemble):
                
                ps = np.max(clf.predict_proba(X), axis=1)
                
                if len(self.prevs[clf_id])>0:
                    # test hipotez
                    stat, p_val = ttest_ind(ps, np.array(self.prevs[clf_id]).flatten())     
                    if p_val<self.alpha:                    
                        _is_drift.append(1)
                        # self.prevs[clf_id] = []
                    else:
                        _is_drift.append(0)
                        self.prevs[clf_id].append(ps)
                else:
                    self.prevs[clf_id].append(ps)
            
            print((_is_drift))
            if np.sum(_is_drift)>(len(_is_drift)/2):
                self.is_drift = True
                self.prevs = [[] for n in range(self.n_clfs)]
            else:
                self.is_drift = False
        else:
            p_vals_sum = []
            for clf_id, clf in enumerate(self.ensemble):
                
                ps = np.max(clf.predict_proba(X), axis=1)

                if len(self.prevs[clf_id])>0:
                    # test hipotez
                    stat, p_val = ttest_ind(ps, np.array(self.prevs[clf_id]).flatten())
                    p_vals_sum.append(p_val)
                    
                self.prevs[clf_id].append(ps)
                
            if len(p_vals_sum)>0:
                aa = np.quantile(p_vals_sum, 0.3)
                print(aa)
                        
                if aa < self.alpha:
                    self.is_drift = True
                    self.prevs = [[] for n in range(self.n_clfs)]
                else:
                    self.is_drift = False
                        
            
                
    def partial_fit(self, X, y, classes):
        
        if self.fitted==True:
            self.detect(X)
            
        if self.is_drift or self.fitted==False:
            for clf in self.ensemble:
                idx = np.random.choice(int(len(y)), size=int((len(y)/2)))
                [clf.partial_fit(X[idx], y[idx], classes) for i in range(self.epochs)]
            self.fitted = True
        return self
    
    def predict(self, X):
        preds = []
        for clf in self.ensemble:
            preds.append(clf.predict(X))
        
        preds = np.array(preds)
        return np.median(preds, axis=0).astype(int)
    
# class certaintyEns2:
#     def __init__(self, base_clf, n_clfs, alpha=0.05, epochs=1):
#         self.n_clfs = n_clfs
#         self.alpha = alpha
#         self.epochs = epochs
        
#         self.ensemble = [clone(base_clf) for n in range(self.n_clfs)]

#         self.is_drift = False
#         self.fitted = False
#         self.prevs = [[] for n in range(self.n_clfs)]    
#         self.weights = np.ones((self.n_clfs))  
        
                
#     def partial_fit(self, X, y, classes):
        
#         if self.fitted==True:
#             for clf_id, clf in enumerate(self.ensemble):
                
#                 ps = np.max(clf.predict_proba(X), axis=1)
                
#                 if len(self.prevs[clf_id])>0:
#                     # test hipotez
#                     stat, p_val = ttest_ind(ps, np.array(self.prevs[clf_id]).flatten())     
#                     if p_val<self.alpha:                    
#                         # drift
#                         [clf.partial_fit(X, y, classes) for i in range(self.epochs)]
#                         print('fit %i' % clf_id)
#                         self.weights[clf_id]+=1
#                         self.prevs[clf_id] = []
                        
#                     else:
#                         self.prevs[clf_id].append(ps)
#                 else:
#                     self.prevs[clf_id].append(ps)

#         else:
#             for clf in self.ensemble:
#                 [clf.partial_fit(X, y, classes) for i in range(self.epochs)]
#             self.fitted = True
            
#         return self
    
#     def predict(self, X):
#         preds = []
#         for clf in self.ensemble:
#             preds.append(clf.predict(X))
        
        # preds = np.array(preds)
        # print(preds.shape)
        
        # weights = np.array((self.weights/np.sum(self.weights)))
        
        # wp = preds*weights
        # print(wp.shape)
        
        # return np.mean(wp, axis=0).astype(int)
    
def get_real_drifts(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts
