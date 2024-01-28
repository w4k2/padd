import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time
from scipy.stats import ttest_ind

## Support nie uśredniony tylko dla wsadu + test hipotez || proby niezal.
## ttest ind



n_chunks = 200
epochs = 150

real_drifts = np.linspace(0,n_chunks,6)[:-1]
real_drifts += (real_drifts[1]/2)


stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=500,
                         n_drifts=5,
                         n_classes=2,
                         n_features=10,
                         n_informative=10,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=5)

clf = MLPClassifier(hidden_layer_sizes=(10,10,10))
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
        [clf.partial_fit(X, y, np.arange(5)) for e in range(epochs)]
    

print(drifts)
p_vals = np.array(p_vals)

s = 0.1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(gaussian_filter1d(res, s), label='BAC', color='tomato', ls=':')
# ax.plot(gaussian_filter1d(res_observed, s), label='BAC observed', color='tomato')
ax.plot(gaussian_filter1d(sup, s), label='MPS', color='cornflowerblue')
ax.scatter(np.arange(len(p_vals)), p_vals, label='p value', color='cornflowerblue', s=10)
ax.vlines(real_drifts,0,1, color='red')
ax.set_xticks(drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')