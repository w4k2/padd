import numpy as np
import matplotlib.pyplot as plt
from utils import mask_to_indexes, dderror, find_real_drift
from scipy.ndimage import gaussian_filter

_n_chunks = 250
_n_drifts = 10

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
 
_alpha = np.linspace(0.03, 0.2, 15)
_th = np.linspace(0.1, 0.3, 10)

reps = 10

res_dets = np.load('res/mini_exp_v2.npy')
print(np.sum(res_dets))

print(res_dets.shape) 
   

n_f_id = 1
n_f = _n_features[n_f_id]

css_id = 1

aa = res_dets[:, n_f_id, css_id]

rr = np.zeros((len(_alpha),len(_th),3))

for a_id in range(len(_alpha)):
    for th_id in range(len(_th)):
        
        reps_errs = []
        for r in range(10):
            this_dets = aa[r, a_id, th_id]
            this_dets_ids = mask_to_indexes(this_dets)
            
            if len(this_dets_ids):
                print(this_dets_ids)
            
            dderr = dderror(find_real_drift(_n_chunks, _n_drifts),this_dets_ids)
            
            reps_errs.append(dderr)
        
        reps_errs = np.array(reps_errs)
        # print(reps_errs)
        # exit()
        reps_errs[np.isinf(reps_errs)] = np.nan
        rr[a_id, th_id] = np.nanmean(reps_errs, axis=0)
        
print(rr.shape) # 15, 10, 3

nanmask = np.isnan(rr)
print(np.sum(nanmask))
# exit()
for c in range(3):
    rr[:,:,c][np.isnan(rr[:,:,c])] = np.nanmax(rr[:,:,c])
    rr[:,:,c] -= np.min(rr[:,:,c])      
    rr[:,:,c] /= np.max(rr[:,:,c]) 
rr[nanmask] = np.nan
            
        
th_id = np.argmin(np.min(np.mean(rr, axis=2), axis=1))
print(th_id, _th[th_id])


a_id = np.argmin(np.min(np.mean(rr, axis=2), axis=0))
print(a_id, _alpha[a_id])

fig, ax = plt.subplots(1,2,figsize=(10,5), sharey=True)

s = 1.2
ax[0].plot(_alpha, gaussian_filter(rr[:,th_id,0],s), label= 'D1', color='red')
ax[0].plot(_alpha, gaussian_filter(rr[:,th_id,1],s), label= 'D2', color='forestgreen')
ax[0].plot(_alpha, gaussian_filter(rr[:,th_id,2],s), label= 'R', color='cornflowerblue')

# ax[0].legend(frameon=False)
ax[0].set_xlabel('alpha')
ax[0].set_ylabel('normalized mean error measure')

ax[0].text(_alpha[0], 1, 'threshold = %0.3f' % _th[th_id])

####

ax[1].plot(_th, gaussian_filter(rr[a_id,:,0],s), label= 'D1', color='red')
ax[1].plot(_th, gaussian_filter(rr[a_id,:,1],s), label= 'D2', color='forestgreen')
ax[1].plot(_th, gaussian_filter(rr[a_id,:,2],s), label= 'R', color='cornflowerblue')

ax[1].legend(frameon=False)
ax[1].set_xlabel('threshold')

ax[1].text(_th[0], 1, 'alpha = %0.3f' % _alpha[a_id])

####

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.grid(ls=':')


plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_exp1/exp0_red.png')
plt.savefig('fig_exp1/exp0_red.eps')

# time.sleep(2)