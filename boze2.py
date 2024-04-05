import numpy as np
from strlearn.streams import StreamGenerator
from methods import CDET
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

np.random.seed(14112)

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

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
               
stream = StreamGenerator(
        n_chunks=_n_chunks,
        chunk_size=_chunk_size,
        n_drifts=10,
        n_classes=2,
        n_features=2,
        n_informative=2,
        n_clusters_per_class=1,
        n_redundant=0,
        n_repeated=0, 
        random_state=1120)

aa=5

d = CDET(ensemble_size=4)

str_probas = []

for chunk_id in range(_n_chunks):
    X, y = stream.get_chunk()
    
    d.process(X)
    
    if chunk_id==0:
        space_x = np.linspace(-aa, aa, 100)
        xx, yy = np.meshgrid(space_x, space_x)
        space = np.vstack((xx.flatten(), yy.flatten())).swapaxes(0,1)
        print(space.shape)
        
        proba = d._predict_proba(space)
        print(proba.shape)
        
    proba_samples = d._predict_proba(X)
    
    m_probas = np.mean(proba_samples, axis=0)
    print(m_probas)
    # exit()
    str_probas.append(m_probas)
    
str_probas = np.array(str_probas)

print(str_probas, str_probas.shape)


fig, ax = plt.subplots(2,4,figsize=(12,6), sharex=True, sharey=True)

for i in range(4):
    ax[0,i].set_title('output %i' % i)

    ax[0,i].scatter(space[:,0], space[:,1], c=proba[:,i], 
                cmap='coolwarm', alpha=1, s=50)

    ax[0,1].grid(ls=':')
    
    ax[0,i].scatter(X[:,0], X[:,1], c='white', marker='x', s=1)

    ax[0,i].set_xlim(-aa,aa)
    ax[0,i].set_ylim(-aa,aa)
    
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])
    ax[1,i].spines['top'].set_visible(False)
    ax[1,i].spines['right'].set_visible(False)

axx = plt.subplot(2,1,2)
for i in range(4):
    axx.plot(str_probas[:,i], color = 'black', ls=['-', ':', '--', '-.'][i], label = 'output %i' % i, lw=1)
    
axx.legend(frameon=False, loc=9, ncol=4)
axx.spines['top'].set_visible(False)
axx.spines['right'].set_visible(False)

drfs = get_real_drfs(_n_chunks, 10)
axx.set_xticks(drfs, drfs.astype(int))
r = 0.0025
axx.set_ylim(np.min(str_probas)-0.3*r, np.max(str_probas)+r)
axx.grid(ls=':')
axx.set_xlim(0,250)

plt.tight_layout()
plt.savefig('foo.png')
# plt.savefig('trash/%04d.png' % chunk_id)
exit()