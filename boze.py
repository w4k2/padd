import numpy as np
from strlearn.streams import SemiSyntheticStreamGenerator
from tqdm import tqdm
from methods import CDET
import os
import matplotlib.pyplot as plt
import time
from scipy.stats import ttest_ind

np.random.seed(9299)

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
        

_n_chunks = 250
_chunk_size = 200
_n_drifts = 20

reps = 5
random_states = np.random.randint(100,10000,reps)

dataset_names = os.listdir('static_data')
try: 
    dataset_names.remove('.DS_Store')
except:
    print(dataset_names)

data = np.loadtxt('static_data/%s' % dataset_names[3], delimiter=',')
X, y = data[:,:-1], data[:,-1]
print(X.shape, y.shape)
                    
stream = SemiSyntheticStreamGenerator(
    X, y,
    n_chunks=_n_chunks,
    chunk_size=_chunk_size,
    n_drifts=_n_drifts,
    n_features=2,
    interpolation='cubic',
    random_state=None)

aa=50

d = CDET(ensemble_size=9)


fig, axx = plt.subplots(3,3,figsize=(10,10), sharex=True, sharey=True)
for chunk_id in range(_n_chunks):
    X, y = stream.get_chunk()
    
    d.process(X)
    
    space_x = np.linspace(-aa, aa, 50)
    xx, yy = np.meshgrid(space_x, space_x)
    space = np.vstack((xx.flatten(), yy.flatten())).swapaxes(0,1)
    print(space.shape)
    
    proba = d._predict_proba(space)
    print(proba.shape)
    
    max_std_s_ids = np.argsort(np.std(proba, axis=0))
    
    ###
    
    for i, ax in enumerate(axx.ravel()):
        ax.scatter(space[:,0], space[:,1], c=proba[:,max_std_s_ids[i]], 
                   cmap='magma', alpha=1, s=50)
        
        ax.scatter(X[:,0], X[:,1], c='white', marker='o', s=1)

        ax.set_xlim(-aa,aa)
        ax.set_ylim(-aa,aa)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('trash/%04d.png' % chunk_id)
    # exit()
    
    ax.cla()
    # time.sleep(0.3)
    
    
    
    