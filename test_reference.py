import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import OneClassDriftDetector
from reference.MD3 import MD3
from methods import CDET


n_chunks= 250
n_drifts = 10

real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
real_drifts += (real_drifts[1]/2)

stream = StreamGenerator(n_chunks=n_chunks,
                        chunk_size=200,
                        n_drifts=n_drifts,
                        n_classes=2,
                        n_features=90,
                        n_informative=30,
                        n_redundant=0,
                        n_repeated=0,
                        concept_sigmoid_spacing=900,
                        random_state=None)

drifts_cddd = []
drifts_ocdd = []
drifts_md3 = []
drifts_ours = []

cddd = CentroidDistanceDriftDetector(sensitive=0.2, distance_p=2, filter_size=3)
ocdd = OneClassDriftDetector(size = 200, dim = 90, percent = 0.99, nu=0.5)
md3 = MD3(sigma=0.15)
cdet = CDET(alpha=0.1, ensemble_size=12, n_replications=12, stat_proba=50, neck_width=10, th=0.23)

for chunk_id in range(n_chunks):
    X, y = stream.get_chunk()
    
    cddd.process(X)
    ocdd.process(X)
    md3.process(X,y)
    cdet.process(X)
    
    if cddd._is_drift == True:
        drifts_cddd.append(chunk_id)
        
    if ocdd._is_drift == True:
        drifts_ocdd.append(chunk_id)
        
    if md3._is_drift == True:
        drifts_md3.append(chunk_id)
        
    if cdet._is_drift == True:
        drifts_ours.append(chunk_id)
    
   
print(drifts_cddd)
print(drifts_ocdd)
print(drifts_md3)
print(drifts_ours)

s = 1
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.vlines(drifts_cddd, 0, 1, color='red', label='cddd')
ax.vlines(drifts_ocdd, 1, 2, color='blue', label='ocdd')
ax.vlines(drifts_md3, 2, 3, color='green', label='md3')
ax.vlines(drifts_ours, 3, 4, color='orange', label='cdet')
ax.set_xticks(real_drifts)

ax.legend()
ax.grid(ls=':')

plt.tight_layout()
plt.savefig('foo.png')