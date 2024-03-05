import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from vapor.methods import certaintyEns, get_real_drifts
from ceADWIN import ceADWIN

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
                         concept_sigmoid_spacing=(5))


adwin = ceADWIN()
clf = MLPClassifier(hidden_layer_sizes=(10,10,10))

detections = []

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()                                     
      
    # test
    if chunk_id>0:
        proba = np.max(clf.predict_proba(X), axis=1)
    
        adwin.feed(X, proba)
    
    # train
   
    try:
        if adwin.isdrift:
            detections.append(chunk_id)
            [clf.partial_fit(X,y, np.arange(5)) for i in range(epochs)]
    except:
        [clf.partial_fit(X,y, np.arange(5)) for i in range(epochs)]
        
fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.vlines(detections, 0, 1, color='red', ls=':')
ax.set_xticks(get_real_drifts(n_chunks, 5))
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')