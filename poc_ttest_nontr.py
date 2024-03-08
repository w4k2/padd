import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind



n_chunks = 500
real_drifts = np.linspace(0,n_chunks,11)[:-1]
real_drifts += (real_drifts[1]/2)

mps_th=0.85
n_cl = 2


stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=300,
                         n_drifts=10,
                         n_classes=n_cl,
                         n_features=25,
                         n_informative=5,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=999)

clf = MLPClassifier(hidden_layer_sizes=(10))
# clf = GaussianNB()

res = []
# res_observed = []
sup = []
p_vals = []

drifts = []
detected = True

prevs = []

alpha=0.02

for chunk_id in range(n_chunks):
    X, _ = stream.get_chunk()
    
    # test
    if chunk_id>0:
        ps = np.max(clf.predict_proba(X), axis=1)    
        
        if len(sup)>0 and len(prevs)>0:
            # test hipotez
            _prevs = np.array(prevs).flatten()
            stat, p_val = ttest_ind(ps, _prevs)     
            p_vals.append(p_val)   
            print(stat,p_val)

            if p_val<alpha:                    
                detected = True
                drifts.append(chunk_id)
                prevs = []
            else:
                detected = False
                prevs.append(ps)

        else:
            detected = False
            prevs.append(ps)
        
        sup.append(np.mean(ps))
        
            
    # train
    print(drifts, detected)
    if chunk_id<1: # detected==True or 
        print('TRAINING', drifts)
        clf.partial_fit(X, np.random.choice(n_cl, X.shape[0]), np.arange(n_cl))
    

print(drifts)
p_vals = np.array(p_vals)

s = 1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(sup, label='MPS', color='cornflowerblue')
ax.vlines(drifts,0,1, color='red', ls=':')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')