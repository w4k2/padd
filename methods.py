import numpy as np
from scipy.stats import ttest_ind

def relu(x):
    return x * (x > 0)   
    
class CDET:
    def __init__(self, alpha=0.05, ensemble_size=30, n_replications=35, stat_proba = 75, neck_width = 10, th = 0.17):
        
        self.alpha=alpha
        self.ensemble_size = ensemble_size      # wielkość zespołu detektorów
        self.n_replications = n_replications    # liczba replikacji testu
        self.stat_proba = stat_proba            # liczba obiektów losowanych do testu
        self.neck_width =  neck_width
        self.th = th
                
        self.past_probas = [[] for _ in range(self.ensemble_size)]
        self._is_drift = None # on init

    def process(self, X):
        
        # Init
        if self._is_drift is None:
            self._is_drift = False
            self.n_features = X.shape[1]
            
            self.stack = [
                np.random.normal(0,0.1,
                                (self.n_features+1,
                                self.neck_width)),
                np.random.normal(0,0.1,
                                (self.neck_width,
                                self.ensemble_size)),
            ]

        self.current_probas = self._predict_proba(X)
    
        if len(self.past_probas[0]) > 0:
            # Dla każdego członka zespołu
            indications = np.zeros((self.ensemble_size, self.n_replications))
            for member_id, (_past, current) in enumerate(zip(self.past_probas,
                                                                self.current_probas.T)):
                # Konkatenujemy przeszłe próbki
                past = np.concatenate(_past)
                
                # Replikujemy pomiar p-value
                for repliaction_id in range(self.n_replications):
                    a = np.random.choice(past, self.stat_proba)
                    b = np.random.choice(current, self.stat_proba)
            
                    stat, pval = ttest_ind(a, b)
                    indications[member_id, repliaction_id] = pval<self.alpha
              
            # print('sum', np.sum(indications))
            th = self.th*self.ensemble_size*self.n_replications
            # print(th, np.sum(indications))

            if np.sum(indications) > th:
                # Indicate drift
                self._is_drift = True
                # print('drf')
                # Reset past probas
                self.past_probas = [[] for _ in range(self.ensemble_size)]
            else:
                self._is_drift = False
        
        # Składowanie wsparć
        for member_id, probas in enumerate(self.current_probas.T):
            self.past_probas[member_id].append(probas)
            
    def _predict_proba(self, X):
        # Copy input
        val = np.concatenate((np.copy(X),
                              np.ones((X.shape[0], 1))), axis=1)
        
        # Propagate through layers
        for layer_id, layer in enumerate(self.stack):
            val = relu(val @ layer)
        
        # # Calculate softmax
        predict_proba = np.exp(val - np.max(val, axis=1)[:,None])
        predict_proba = predict_proba / np.sum(predict_proba, axis=1)[:,None]

        return predict_proba
        
            
        
        
        








# prev version
# class CDET:
#     def __init__(self, base_mlp, n_epochs, alpha, mps_th, balanced=False, psk=True):
#         self.base_mlp = clone(base_mlp)
#         self.n_epochs = n_epochs
#         self.alpha = alpha
#         self.balanced = balanced
#         self.psk = psk
#         self.mps_th = mps_th
    
#         self.is_drift = None
        
#     def process(self, X, y):
        
#         #setup on 1st chunk
#         if self.is_drift is None:
#             self.prevs = []
#             self.is_drift = False

#             self._train(X, y)

#         else:
#             ps = np.max(self.base_mlp.predict_proba(X), axis=1)    
#             if len(self.prevs)>0:
                
#                 _prevs = np.array(self.prevs).flatten()
                
#                 # test hipotez
#                 if self.balanced:
#                     _prevs = np.random.choice(_prevs, len(ps), replace=False)
                
#                 _, p_val = ttest_ind(ps, _prevs)
#                 # print(p_val)

#                 if p_val<self.alpha: 
#                     print(p_val)
#                     self.is_drift = True
#                     self.prevs = []
                    
#                 else:
#                     self.is_drift = False
#                     self.prevs.append(ps)
                    
#             else:
#                 self.is_drift = False
#                 self.prevs.append(ps)

#         if self.is_drift:
#             self._train(X, y)
            
        
#     def _train(self, X, y):
#         if self.psk:
#             counter=0
#             while(1):
#                 counter+=1
#                 self.base_mlp.partial_fit(X, y, np.arange(2))
#                 mps = np.mean(np.max(self.base_mlp.predict_proba(X), axis=1))
#                 if mps>=self.mps_th:
#                     break
#                 if counter>100000: #zapobiegawczo
#                     break
#         else:
#             [self.base_mlp.partial_fit(X, y, np.unique(y)) for e in range(self.n_epochs)]
            