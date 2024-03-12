import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
from scipy.stats import ttest_ind


## Klasyfikator szkolony do określonego MPS


n_chunks = 500
real_drifts = np.linspace(0,n_chunks,11)[:-1]
real_drifts += (real_drifts[1]/2)

mps_th=0.8


stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=200,
                         n_drifts=10,
                         n_classes=2,
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

alpha=0.05

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    
    # test
    if chunk_id>0:
        ps = np.max(clf.predict_proba(X), axis=1)    
        bac = balanced_accuracy_score(y, clf.predict(X))
        
        if len(sup)>0:
            # test hipotez
            stat, p_val = ttest_ind(ps, np.array(prevs).flatten())     
            p_vals.append(p_val)   
            print(stat,p_val)

            if p_val<alpha:                    
                detected = True
                drifts.append(chunk_id)
                prevs = []
            else:
                detected = False
                prevs.append(ps)
     
        res.append(bac)
        # res_observed.append(bac_observed)
        sup.append(np.mean(ps))
        
        # time.sleep(1)
        
    # train
    print(drifts, detected)
    if detected==True or chunk_id<1:
        print('TRAINING', drifts)
        while(1):
            clf.partial_fit(X, y, np.arange(2))
            mps = np.mean(np.max(clf.predict_proba(X), axis=1))
            print(mps)
            if mps>=mps_th:
                break
    

print(drifts)
p_vals = np.array(p_vals)

s = 1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(gaussian_filter1d(res, s), label='BAC', color='tomato', ls=':')
ax.plot(sup, label='MPS', color='cornflowerblue')
# ax.scatter(np.arange(len(p_vals)), p_vals, label='p value', color='cornflowerblue', s=10)
ax.vlines(drifts,0,1, color='red', ls=':')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')