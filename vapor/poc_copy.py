import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from methods import RandomSight
from time import sleep
np.set_printoptions(precision=3)
from tqdm import tqdm


"""
- centyle w ttest
- mlp -- albo dużo mlp albo jeden dyży 
- mlp oszukać ze umie

opcjonalnie /  alternatywne:
- etykiety z klasteryzacji

"""

n_chunks = 200
n_drifts = 10
ensemble_size = 10      # wielkość zespołu detektorów
n_replications = 10     # liczba replikacji testu
stat_proba = 500        # liczba obiektów losowanych do testu

real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
real_drifts += (real_drifts[1]/2)

n_cl = 2

stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=300,
                         n_drifts=n_drifts,
                         n_classes=n_cl,
                         n_features=25,
                         n_informative=25,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=999,
                         random_state=1410)

model = RandomSight(n_features=25, n_outputs=ensemble_size)

drifts = []
detected = True

past_probas = [[] for _ in range(ensemble_size)]
alpha = 0.01

psmin, psmax = 1, 0

p_history = [[] for _ in range(ensemble_size)]

for chunk_id in tqdm(range(n_chunks)):
    X, _ = stream.get_chunk()
    
    # Pobieranie wsparć
    current_probas = model.predict_proba(X)

    # Analiza
    # Tylko jeżeli mamy już jakieś wsparcia historyczne
    if len(past_probas[0]) > 0:
        # Dla każdego członka zespołu
        # print(ensemble_size, len(past_probas), current_probas.shape)
        indications = np.zeros(ensemble_size)
        for member_id, (_past, current) in enumerate(zip(past_probas,
                                                               current_probas.T)):
            # Konkatenujemy przeszłe próbki
            past = np.concatenate(_past)
            
            # Replikujemy pomiar p-value
            pvals = []
            for i in range(n_replications):
                a = np.random.choice(past, stat_proba)
                b = np.random.choice(current, stat_proba)
        
                stat, pval = ttest_ind(a, b)
                pvals.append(pval)
            
            # Zapiszmy ustabilizowaną p-wartość do prezentacji
            p_history[member_id].append(np.mean(pvals))
            
            # Jeżeli różnica jest istotna
            if np.mean(pvals) < alpha:
                indications[member_id] = 1
                
            # print('M', member_id, past.shape, current.shape)
        if np.sum(indications) > 0:
            # Indicate drift
            drifts.append(chunk_id)
            
            # Reset past probas
            past_probas = [[] for _ in range(ensemble_size)]
    
    # Składowanie wsparć
    for member_id, probas in enumerate(current_probas.T):
        past_probas[member_id].append(probas)

print(drifts)
p_history = np.array(p_history)
print(p_history.shape)

s = 1
fig, ax = plt.subplots(1,1,figsize=(5,5))
# ax.plot(sup, label='MPS', color='cornflowerblue')
ax.plot(p_history.T, c='black', alpha=1, lw=1)
ax.vlines(drifts,0,1, color='red', ls=':')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')