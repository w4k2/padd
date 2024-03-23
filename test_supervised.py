import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from sklearn.naive_bayes import GaussianNB
from detectors.adwin import ADWIN
from detectors.ddm import DDM
from detectors.eddm import EDDM
from detectors.meta import MetaClassifier
"""
 W eksperymentach zosatwiamy wartości domyślne z multiflow!

"""
n_chunks = 250
n_drifts=5
_chunk_size = 200
_n_informative_fraction = 0.3

real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
real_drifts += (real_drifts[1]/2)

_n_features = [30, 60, 90]
ii = 0

stream = StreamGenerator(
                        n_chunks=n_chunks,
                        chunk_size=_chunk_size,
                        n_drifts=n_drifts,
                        n_classes=2,
                        n_features=_n_features[ii],
                        n_informative=int(_n_informative_fraction * _n_features[ii]),
                        n_redundant=0,
                        n_repeated=0, 
                        concept_sigmoid_spacing=999,
                        random_state=787)


drifts_adwin = []
drifts_ddm = []
drifts_eddm = []

adwin = MetaClassifier(base_clf=GaussianNB(), detector=ADWIN(delta=0.007))
ddm = MetaClassifier(base_clf=GaussianNB(), detector=DDM(drift_lvl=3.5))
eddm = MetaClassifier(base_clf=GaussianNB(), detector=EDDM(drift_lvl=0.85))

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    
    for d in [adwin, ddm, eddm]:
        if chunk_id==0:
            # train
            d.partial_fit(X, y, np.unique(y))
        else:
            d.predict(X)
            d.partial_fit(X, y, np.unique(y))
            
    if chunk_id>0:
        if adwin.detector.drift[-1]==2:
            drifts_adwin.append(chunk_id)
        
        if ddm.detector.drift[-1]==2:
            drifts_ddm.append(chunk_id)
            
        if eddm.detector.drift[-1]==2:
            drifts_eddm.append(chunk_id)
    
    
   
print(drifts_adwin)
print(drifts_ddm)
print(drifts_eddm)

s = 1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.vlines(drifts_adwin, 0, 1, color='red', label='drifts_adwin')
ax.vlines(drifts_ddm, 1, 2, color='blue', label='drifts_ddm')
ax.vlines(drifts_eddm, 2, 3, color='green', label='drifts_eddm')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')