import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import mask_to_indexes, dderror, find_real_drift
import os

_n_chunks = 250
_n_drifts = 10

_n_features = [30, 60, 90]
 
_alpha = np.linspace(0.000001, 0.05, 15)
_th = np.linspace(0.000005, 0.1, 10)

reps = 5
random_states = np.random.randint(100,10000,reps)

dataset_names = os.listdir('static_data')
try: 
    dataset_names.remove('.DS_Store')
except:
    print(dataset_names)

res_dets = np.load('res/exp0_semi.npy')
print(res_dets.shape)
   
fig, ax = plt.subplots(3,4,figsize=(25,20), sharex=True, sharey=True)

for n_f_id, n_f in enumerate(_n_features):
    for data_id, data_name in enumerate(dataset_names):
        
        aa = res_dets[:, data_id, n_f_id]
        
        rr = np.zeros((len(_alpha),len(_th),3))
        
        for a_id in range(len(_alpha)):
            for th_id in range(len(_th)):
                
                reps_errs = []
                for r in range(reps):
                    this_dets = aa[r, a_id, th_id]
                    this_dets_ids = mask_to_indexes(this_dets)
                    
                    # if len(this_dets_ids):
                    #     print(this_dets_ids)
                    
                    dderr = dderror(find_real_drift(_n_chunks, _n_drifts),this_dets_ids)
                    
                    reps_errs.append(dderr)
                
                reps_errs = np.array(reps_errs)
                # print(reps_errs)
                # exit()
                reps_errs[np.isinf(reps_errs)] = np.nan
                # print(np.nanmean(reps_errs, axis=0))
                rr[a_id, th_id] = np.nanmean(reps_errs, axis=0)
                
        for c in range(3):
            rr[:,:,c][np.isnan(rr[:,:,c])] = np.nanmax(rr[:,:,c])
            rr[:,:,c] -= np.min(rr[:,:,c])      
            rr[:,:,c] /= np.max(rr[:,:,c]) 
        # print(rr)  
        # exit()
        ax[n_f_id, data_id].imshow(rr[:,:,:], aspect='auto', cmap='coolwarm', interpolation='none')
        ax[n_f_id, data_id].set_title('F:%i | %s' % (n_f, data_name))
        ax[n_f_id, data_id].set_yticks(np.arange(len(_alpha)), ['%.3f' % a for a in _alpha])
        ax[n_f_id, data_id].set_xticks(np.arange(len(_th)), ['%.3f' % a for a in _th])
        
        ax[n_f_id, data_id].set_ylabel('alpha')
        ax[n_f_id, data_id].set_xlabel('threshold')
       
        rr_m = np.mean(rr, axis=2)
        for _a in np.arange(len(_alpha)):
            for _b in np.arange(len(_th)):
                ax[n_f_id, data_id].text(_b, _a, "%.2f" % (
                    rr_m[_a, _b]
                    ) , va='center', ha='center', 
                        c='white' if rr_m[_a, _b] < 0.5 
                        else 'black', fontsize=11)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig_exp1/exp0_semisyn.png')

# time.sleep(2)