import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

n_chunks = 200
epochs = 150

stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=400,
                         n_drifts=5,
                         n_classes=5,
                         n_features=30,
                         n_informative=30,
                         n_redundant=0,
                         n_repeated=0)

clf = MLPClassifier(hidden_layer_sizes=(100,100,100))
res = []
# res_observed = []
sup = []

drifts = []
detected = True

prevs = []

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    
    # labels_ids = np.random.choice(len(y), int(L_access*len(y)), replace=False)
                                     
    
    # test
    if chunk_id>0:
        mps = np.mean(np.max(clf.predict_proba(X), axis=1))      
        bac = balanced_accuracy_score(y, clf.predict(X))
        # bac_observed = balanced_accuracy_score(y[labels_ids], clf.predict(X[labels_ids]))
        # print(chunk_id, bac, bac_observed, mps)

        
        if len(sup)>0:
                       
            if (mps > np.mean(prevs) + 3*np.std(prevs) \
                or mps < np.mean(prevs) - 3*np.std(prevs)) \
                and len(prevs)>5:
                    
                detected=True
                drifts.append(chunk_id)
                prevs = []
            else:
                detected=False
                prevs.append(mps)
     
        res.append(bac)
        # res_observed.append(bac_observed)
        sup.append(mps)
        
        # time.sleep(1)
        
    # train
    print(drifts, detected)
    if detected==True:
        print('TRAINING', drifts)
        [clf.partial_fit(X, y, np.arange(5)) for e in range(epochs)]
    

print(drifts)
s = 0.1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(gaussian_filter1d(res, s), label='BAC', color='tomato', ls=':')
# ax.plot(gaussian_filter1d(res_observed, s), label='BAC observed', color='tomato')
ax.plot(gaussian_filter1d(sup, s), label='MPS', color='cornflowerblue')
ax.set_xticks(drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')