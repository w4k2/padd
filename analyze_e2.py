import numpy as np
import matplotlib.pyplot as plt
from utils import dderror, mask_to_indexes

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

_n_chunks = 250

_n_features = [30, 60, 90]
_n_drifts = [3, 5, 10, 15]
 
reps = 10
n_methods = 4

res_dets = np.load('res/exp1_comp_final.npy') # replications, features, concept sigmoid, detectors, chunks

results = np.zeros((10,3,2,4,7,3))
str_names = np.zeros((3,2,4)).astype('object')

for n_f_id, n_f in enumerate(_n_features):
    for css_id, css in enumerate(['GRAD', 'SUDD']):
        for n_d_id, n_d in enumerate(_n_drifts):
            str_names[n_f_id, css_id, n_d_id] = ('F: %02d | %s | D: %02d' % (n_f, css, n_d))
            
            for r in range(reps):
                for method_id in range(7):
                    
                    dets = mask_to_indexes(res_dets[r, n_f_id, css_id, n_d_id, method_id])
                    drifts = get_real_drfs(_n_chunks, n_d).astype(int)
                    errs = dderror(drifts, dets)
                    # print(errs)
                    results[r, n_f_id, css_id, n_d_id, method_id] = errs                

mean_results = np.mean(results, axis=0)
print(mean_results.shape)

mean_results = mean_results.swapaxes(0,1).reshape(-1,7,3)
print(mean_results.shape)

str_names = str_names.swapaxes(0,1).reshape(-1)

fig, ax = plt.subplots(1, 3, figsize=(15,8), sharex=True, sharey=True, dpi=100)

mr = mean_results[:,:,0]
mr[np.isnan(mr)] = 15
ax[0].imshow(mean_results[:,:,0], cmap='coolwarm', aspect='auto', vmin=0, vmax=30)
ax[0].set_title('D1 - Detection from nearest drift', fontsize=15)

ax[1].imshow(mean_results[:,:,1], cmap='coolwarm', aspect='auto', vmin=0, vmax=30)
ax[1].set_title('D2 - Drift from nearest detection', fontsize=15)

ax[2].imshow(mean_results[:,:,2], cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
ax[2].set_title('R - Drifts to detections ratio', fontsize=15)

for aa in ax:
    aa.set_xticks(np.arange(7), ['MD3', 'OCDD', 'CDDD', 'PADD', 'ADWIN', 'DDM', 'EDDM'], rotation=90)
    aa.set_yticks(np.arange(24), str_names)
    
for i in range(3):
    for _a in range(24):
        for _b in range(7):
            if np.isnan( mean_results[:,:,i][_a, _b]) == False:
                ax[i].text(_b, _a, "%.3f" % (
                    mean_results[:,:,i][_a, _b]
                    ) , va='center', ha='center', 
                        c='white' 
                        if (
                        i==0 and 
                          (mean_results[:,:,i][_a, _b] > 30 or mean_results[:,:,i][_a, _b] < 3))
                        or (
                        i==1 and 
                          (mean_results[:,:,i][_a, _b] > 80 or mean_results[:,:,i][_a, _b] < 3))
                        or (
                        i==2 and
                          (mean_results[:,:,i][_a, _b] > 0.9 or mean_results[:,:,i][_a, _b] < .07))
                        else 'black', 
                        fontsize=11)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/errs_nans.png')
plt.savefig('fig/errs_nans.eps')
