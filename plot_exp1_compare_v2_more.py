import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

_n_chunks = 250
_n_drifts = 10

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
_n_drifts = [3, 5, 10, 15]
 
alphas = [0.13, 0.07] # for css 5, 999
ths = [0.26, 0.19] # for css 5, 999

reps = 10
n_methods = 4

res_dets = np.load('res/exp1_comp_v2_more.npy') # replications, features, concept sigmoid, detectors, chunks
print(res_dets.shape)

cm = plt.cm.binary(np.linspace(0,1,256))
print(cm[-1], cm[0])
cm[-1] = [1.,0.,0.,1.]

cm = matplotlib.colors.LinearSegmentedColormap.from_list('colormap', cm)

res_dets -= 0.2
res_dets[:,:,:,:,3][res_dets[:,:,:,:,3]==0.8] = 1

for n_d_id, n_d in enumerate(_n_drifts):
    fig, ax = plt.subplots(len(_n_features), len(_concept_sigmoid_spacing), figsize=(10,10), sharex=True, sharey=True)
    plt.suptitle('%i drifts' % n_d, fontsize=15)

    for n_f_id, n_f in enumerate(_n_features):
        for css_id, css in enumerate(_concept_sigmoid_spacing):
                    
            aa = res_dets[:, n_f_id, css_id, n_d_id].swapaxes(0,1).reshape(-1,_n_chunks)
            ax[n_f_id, css_id].imshow(aa, cmap=cm, aspect='auto', interpolation='none')
            ax[n_f_id, css_id].set_title('CSS:%i | F:%i' % (css, n_f))
            ax[n_f_id, css_id].set_xticks(get_real_drfs(_n_chunks, n_d), ['%i' % a for a in get_real_drfs(_n_chunks, n_d)], rotation=90)
            ax[n_f_id, css_id].set_yticks(np.arange(5,75,10),['MD3', 'OC', 'CD', 'CDET', 'ADWIN', 'DDM', 'EDDM'])
            ax[n_f_id, css_id].hlines(np.arange(10,80,10), 0, _n_chunks, lw=1, ls=':', color='gray')
            ax[n_f_id, css_id].grid(ls=':')
            ax[n_f_id, css_id].spines['top'].set_visible(False)
            ax[n_f_id, css_id].spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('foo.png')
            plt.savefig('fig_exp1/exp1_all_%id.png' % n_d)
            # exit()                    