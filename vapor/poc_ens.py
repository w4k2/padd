import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

n_chunks = 200
epochs = 15

stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=400,
                         n_drifts=5,
                         n_classes=5,
                         n_features=30,
                         n_informative=30,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=(50))

n_clfs = 5

ens = [
    MLPClassifier(hidden_layer_sizes=(100))
    for _ in range(n_clfs)
]

res = []
sup = []
stds = []

drifts = []
detected = True

prevs = []

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()                                     
    
    # test
    if chunk_id>0:
        mps = [np.mean(np.max(clf.predict_proba(X), axis=1)) for clf in ens]
        mps_std = np.std(mps)  
        
        aa = np.mean([clf.predict(X) for clf in ens], axis=0)
        # print(aa, aa.shape)
        preds = np.rint(aa)
        bac = balanced_accuracy_score(y, preds)

        print(mps_std)
        stds.append(mps_std)
        
        if len(sup)>0:
                       
            if (mps_std > np.mean(prevs) + 3*np.std(prevs) \
                or mps_std < np.mean(prevs) - 3*np.std(prevs)) \
                and len(prevs)>5:
                    
                detected=True
                drifts.append(chunk_id)
                prevs = []
            else:
                detected=False
                prevs.append(mps_std)
     
        res.append(bac)
        # res_observed.append(bac_observed)
        sup.append(mps)
        
        # time.sleep(1)
        
    # train
    print(drifts, detected)
    if detected==True:
        print('TRAINING', drifts)
        for clf in ens:
            [clf.partial_fit(X, y, np.arange(5)) for e in range(epochs)]
    

print(drifts)
s = 0.1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(gaussian_filter1d(res, s), label='BAC', color='tomato', ls=':')
ax.plot(gaussian_filter1d(sup, s), label='MPS', color='cornflowerblue')
ax.plot(gaussian_filter1d(stds, s), label='std', color='gold')
ax.set_xticks(drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')