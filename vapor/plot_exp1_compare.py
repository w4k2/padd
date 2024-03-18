import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts


_n_chunks = 250

_n_drifts = [3, 5, 10, 15]
_chunk_size = [100, 250, 500]
_n_features = [20, 40, 80, 120]
_concept_sigmoid_spacing = [5, 999]
_n_classes = [2,5,10]

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_n_drifts)
            *len(_chunk_size)*len(_n_features)
            *len(_concept_sigmoid_spacing)*len(_n_classes)*_n_chunks)

n_methods = 6
res_dets = np.load('res/exp1_comp.npy') # replications, drifts, chunk size, features, concept sigmoid, classes, detections, chunks

print(res_dets.shape)
for conc_ss_id, conc_ss in enumerate(_concept_sigmoid_spacing):
    for ch_s_id, ch_s in enumerate(_chunk_size):
        for n_cl_id, n_cl in enumerate(_n_classes):
            
            temp = res_dets[:,:,ch_s_id,:,conc_ss_id,n_cl_id]

            fig, ax = plt.subplots(len(_n_drifts), len(_n_features), figsize=(20,10))
            plt.suptitle('CSS: %i | CH_S: %i | CL: %i' % (conc_ss, ch_s, n_cl))
            
            for n_d_id, n_d in enumerate(_n_drifts):            
                for n_f_id, n_f in enumerate(_n_features):
                    
                    aa = temp[:,n_d_id, n_f_id].swapaxes(0,1).reshape(-1,_n_chunks)
                    ax[n_d_id, n_f_id].imshow(aa, cmap='binary', aspect='auto')
                    ax[n_d_id, n_f_id].set_title('D:%i | F:%i' % (n_d, n_f))
                    ax[n_d_id, n_f_id].set_xticks(get_real_drfs(_n_chunks, n_d), ['%i' % a for a in get_real_drfs(_n_chunks, n_d)], rotation=90)
                    ax[n_d_id, n_f_id].set_yticks([5,15,25,35,45,55],['MD3', 'OC', 'CD', 'CDETis', 'CDET', 'CDETs'])
                    ax[n_d_id, n_f_id].grid(ls=':')
                    ax[n_d_id, n_f_id].spines['top'].set_visible(False)
                    ax[n_d_id, n_f_id].spines['right'].set_visible(False)
                    
            plt.tight_layout()
            plt.savefig('foo.png')
            plt.savefig('fig_uns/e1_css%i_cs%i_c%i.png' % (conc_ss, ch_s, n_cl))
            # exit()                    