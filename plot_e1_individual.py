import numpy as np
import matplotlib.pyplot as plt

from utils import mask_to_indexes, dderror, find_real_drift

_n_chunks = 250
_n_drifts = 10
 
_alpha = np.linspace(0.03, 0.2, 15)
_th = np.linspace(0.1, 0.3, 10)

reps = 10

res_dets = np.load('res/mini_exp_v2.npy')
print(np.sum(res_dets))

print(res_dets.shape) 
   
fig, ax = plt.subplots(1,3,figsize=(10,5.5), sharex=True, sharey=True)

aa = res_dets[:, 0, 1]

rr = np.zeros((len(_alpha),len(_th),3)).astype(float)

for a_id in range(len(_alpha)):
    for th_id in range(len(_th)):
        
        reps_errs = []
        for r in range(10):
            this_dets = aa[r, a_id, th_id]
            this_dets_ids = mask_to_indexes(this_dets)
            
            if len(this_dets_ids):
                print(this_dets_ids)
            
            dderr = dderror(find_real_drift(_n_chunks, _n_drifts), this_dets_ids, _n_chunks)            
            reps_errs.append(dderr)
        
        reps_errs = np.array(reps_errs)
        rr[a_id, th_id] = np.mean(reps_errs, axis=0)

ax[0].imshow(rr[:,:,0], cmap='bone')
ax[1].imshow(rr[:,:,1], cmap='bone')
ax[2].imshow(rr[:,:,2], cmap='bone')

ax[0].set_title('D1 - Detection from nearest drift')
ax[1].set_title('D2 - Drift from nearest detection')
ax[2].set_title('R - Drifts to detections ratio')

for i in range(3):
    ax[i].set_yticks(np.arange(len(_alpha)), ['%.2f' % a for a in _alpha])
    ax[i].set_xticks(np.arange(len(_th))[::2], ['%.2f' % a for a in _th][::2])
    ax[i].set_xlabel('threshold')

ax[0].set_ylabel('alpha')

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/e0_individual.png')
plt.savefig('fig/e0_individual.eps')
