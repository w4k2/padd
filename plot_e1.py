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
   
fig, ax = plt.subplots(3,2,figsize=(10,10), sharex=True, sharey=True)

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
                    
                    # print(find_real_drift(_n_chunks, _n_drifts), this_dets_ids, _n_chunks)
                    dderr = dderror(find_real_drift(_n_chunks, _n_drifts), this_dets_ids, _n_chunks)
                    
                    reps_errs.append(dderr)
                
                reps_errs = np.array(reps_errs)
                # print(reps_errs)
                # exit()
                # reps_errs[np.isinf(reps_errs)] = np.nan
                # print(np.nanmean(reps_errs, axis=0))
                rr[a_id, th_id] = np.mean(reps_errs, axis=0)
        
        
        for c in range(3):
            rr[:,:,c] -= np.min(rr[:,:,c])      
            rr[:,:,c] /= np.max(rr[:,:,c]) 
        # print(rr)  
        # exit()
        ax[n_f_id, 1-css_id].imshow(rr[:,:,:], aspect='auto', cmap='coolwarm')
        ax[n_f_id, 1-css_id].set_title('%i dim | %s' % (n_f, 'sudden' if css==999 else 'gradual'))
        ax[n_f_id, 1-css_id].set_yticks(np.arange(len(_alpha)), ['%.2f' % a for a in _alpha])
        ax[n_f_id, 1-css_id].set_xticks(np.arange(len(_th)), ['%.2f' % a for a in _th])
        
        if css_id==0:
            ax[n_f_id, 1-css_id].set_ylabel('alpha')
        if n_f_id==2:
            ax[n_f_id, 1-css_id].set_xlabel('threshold')
       
        # rr_m = np.mean(rr, axis=2)
        # for _a in np.arange(len(_alpha)):
        #     for _b in np.arange(len(_th)):
        #         ax[n_f_id, css_id].text(_b, _a, "%.2f" % (
        #             rr_m[_a, _b]
        #             ) , va='center', ha='center', 
        #                 c='white' if rr_m[_a, _b] < 0.5 
        #                 else 'black', fontsize=11)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/exp0.png')
plt.savefig('fig/exp0.eps')
