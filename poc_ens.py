import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from methods import certaintyEns, get_real_drifts
from skmultiflow.meta import LeverageBagging

n_chunks = 200
epochs = 15

stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=400,
                         n_drifts=5,
                         n_classes=3,
                         n_features=30,
                         n_informative=30,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=(50))


ens = certaintyEns(base_clf=MLPClassifier(), n_clfs=19, epochs=200, alpha=0.15)
# lbc = LeverageBagging()

bac_ens = []
# bac_lbc = []
detections = []

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()                                     
    
            
    # test
    if chunk_id>0:
        pred = ens.predict(X)
        bac_ens.append(balanced_accuracy_score(y, pred))
        
        # pred = lbc.predict(X)
        # bac_lbc.append(balanced_accuracy_score(y, pred))
    
    # train
    ens.partial_fit(X, y, np.arange(5))
    # lbc.partial_fit(X, y, np.arange(5))
    
    if ens.is_drift:
        detections.append(chunk_id)
        
fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.plot(bac_ens)
# ax.plot(bac_lbc)
ax.vlines(detections, 0, 1, color='red', ls=':')
ax.set_xticks(get_real_drifts(n_chunks, 5))
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')