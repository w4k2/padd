import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

n_chunks = 100
epochs = 5

stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=200,
                         n_drifts=5,
                         n_classes=2,
                         n_features=10,
                         n_informative=10,
                         n_redundant=0,
                         n_repeated=0)

clf = MLPClassifier()
res = []
sup = []

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    
    # test
    if chunk_id>0:
        mps = np.mean(np.max(clf.predict_proba(X), axis=1))      
        bac = balanced_accuracy_score(y, clf.predict(X))
        print(chunk_id, bac, mps)
        res.append(bac)
        sup.append(mps)
        
    # train
    [clf.fit(X, y) for e in range(epochs)]
    

s = 0.5
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(gaussian_filter1d(res, s), label='BAC', color='tomato')
ax.plot(gaussian_filter1d(sup, s), label='MPS', color='cornflowerblue')

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')