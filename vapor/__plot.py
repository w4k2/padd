import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

_n_chunks = 500

_alpha = np.linspace(0.001, 0.05, 10)
_ensemble_sizes = [5,20,30]
_replications = [5,15,25,35]
_stat_probas = [75,200]
_neck_width = [256, 512]

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_alpha))


res_dets = np.load('res/mini_exp.npy')

print(res_dets.shape)                    
                                    
for sp_id, sp in enumerate(_stat_probas):
    for n_id, n in enumerate(_neck_width):
                                                        
        fig, ax = plt.subplots(1,1,figsize=(15,9), sharex=True, sharey=True)
        plt.suptitle('stat_proba: %i | neck: %i' % (sp, n))


        aa = res_dets[:, -1, 0, sp_id, n_id].swapaxes(0,1).reshape(-1,_n_chunks)
        ax.imshow(aa, cmap='twilight', aspect='auto')

        ax.set_yticks(np.linspace(5,95,10), ['%.3f' % a for a in _alpha])
            
        plt.tight_layout()
        plt.savefig('foo.png')
        # time.sleep(2)