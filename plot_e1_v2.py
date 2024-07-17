import numpy as np
import matplotlib.pyplot as plt
from utils import mask_to_indexes, dderror, find_real_drift

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
   
fig, ax = plt.subplots(2,3,figsize=(10,10), sharex=True, sharey=True)

for n_f_id, n_f in enumerate(_n_features):
    for css_id, css in enumerate(_concept_sigmoid_spacing):
        
        aa = res_dets[:, n_f_id, css_id]
        
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
                reps_errs[np.isinf(reps_errs)] = np.nan
                print(np.nanmean(reps_errs, axis=0))
                rr[a_id, th_id] = np.nanmean(reps_errs, axis=0)

                # reps_errs[np.isinf(reps_errs)] = np.nan
                # print(np.nanmean(reps_errs, axis=0))
                rr[a_id, th_id] = np.mean(reps_errs, axis=0)


        for c in range(3):
            rr[:,:,c][np.isnan(rr[:,:,c])] = np.nanmax(rr[:,:,c])
            rr[:,:,c] -= np.min(rr[:,:,c])      
            rr[:,:,c] /= np.max(rr[:,:,c]) 

        ax[1-css_id, n_f_id].imshow(rr)
        ax[1-css_id, n_f_id].set_title('%i features | %s' % (n_f, 'sudden' if css==999 else 'gradual'))
        ax[1-css_id, n_f_id].set_yticks(np.arange(len(_alpha)), ['%.2f' % a for a in _alpha])
        ax[1-css_id, n_f_id].set_xticks(np.arange(len(_th))[::2], ['%.2f' % a for a in _th][::2])
        
        if n_f_id==0:
            ax[css_id, n_f_id].set_ylabel('alpha')
        if css_id==1:
            ax[css_id, n_f_id].set_xlabel('threshold')
       
plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/exp0.png')
plt.savefig('fig/exp0.eps')
