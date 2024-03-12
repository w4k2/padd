import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import 

n_chunks = 300
n_drifts = 5

real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
real_drifts += (real_drifts[1]/2)

n_cl = 2

stream = StreamGenerator(n_chunks=n_chunks,
                         chunk_size=200,
                         n_drifts=n_drifts,
                         n_classes=n_cl,
                         n_features=40,
                         n_informative=7,
                         n_redundant=0,
                         n_repeated=0, 
                         concept_sigmoid_spacing=500,
                         random_state=876)

drifts_cddd = []

cddd = CentroidDistanceDriftDetector()
ocdd = one

for chunk_id in range(n_chunks):
    X, _ = stream.get_chunk()
    
    cddd.process(X)
    
    if cddd._is_drift == True:
        drifts_cddd.append(chunk_id)
    
   
print(drifts_cddd)

s = 1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.vlines(drifts_cddd, 0, 1, color='red', ls=':', label='cddd')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')