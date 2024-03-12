import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import OneClassDriftDetector
from reference.MD3 import MD3

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
drifts_ocdd = []
drifts_md3 = []

cddd = CentroidDistanceDriftDetector()
ocdd = OneClassDriftDetector(size = 100, dim = 40, percent = 0.85, nu=0.5)
md3 = MD3()

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    
    cddd.process(X)
    ocdd.process(X)
    md3.process(X,y)
    
    if cddd._is_drift == True:
        drifts_cddd.append(chunk_id)
        
    if ocdd._is_drift_chunk == True:
        drifts_ocdd.append(chunk_id)
        
    if md3._is_drift == True:
        drifts_md3.append(chunk_id)
    
   
print(drifts_cddd)
print(drifts_ocdd)
print(drifts_md3)

s = 1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.vlines(drifts_cddd, 0, 1, color='red', ls=':', label='cddd')
ax.vlines(drifts_ocdd, 0, 1, color='blue', ls=':', label='ocdd')
ax.vlines(drifts_md3, 0, 1, color='green', ls=':', label='md3')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')