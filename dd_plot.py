
import numpy as np
import matplotlib.pyplot as plt
from methods import get_real_drifts

_n_chunks = [200, 400]
_chunk_size = [100, 200, 400]
_n_drifts = [3,5,10]
_n_features = [10,20,30]
_n_classes = [2,5,10]
_concept_sigmoid_spacing = [5,999]

_alpha = [.02, .03, .05, .07]


for n_chunks_id, n_chunks in enumerate(_n_chunks):
    res = np.load('res/dets_chunks_%i.npy' % n_chunks)
    
    for chunk_size_id, chunk_size in enumerate(_chunk_size):
        for css_id, css in enumerate(_concept_sigmoid_spacing):
            for n_d_id, nd in enumerate(_n_drifts):
                
                real_drifts = get_real_drifts(n_chunks, nd).astype(int)

                fig, ax = plt.subplots(3,3, figsize=(10*(n_chunks_id+1),5), sharex=True, sharey=True)
                plt.suptitle('CH:%i | S:%i | CSS:%i | D:%i' % (n_chunks, chunk_size, css, nd))
                
                for n_cl_id, n_cl in enumerate(_n_classes):
                    for n_f_id, n_f in enumerate(_n_features):
                
                        temp_res = res[:,chunk_size_id, n_d_id, n_f_id, n_cl_id, css_id]
                        temp_res = temp_res.swapaxes(1,0).reshape(-1,n_chunks)
                        
                        ax[n_cl_id, n_f_id].imshow(temp_res, cmap='binary', interpolation=None)
                        ax[n_cl_id, n_f_id].set_title('CL:%i | F:%i' % (n_cl, n_f))
                        ax[n_cl_id, n_f_id].set_xticks(real_drifts)
                        ax[n_cl_id, n_f_id].set_yticks([10,20,30,40], _alpha)
                        ax[n_cl_id, n_f_id].grid(ls=':')
                        
                plt.tight_layout()
                plt.savefig('foo.png')
                plt.savefig('fig/ch%i_s%i_css%i_d%i.png' % (n_chunks, chunk_size, css, nd))
                # exit()

