import numpy as np
import matplotlib.pyplot as plt

_n_chunks = 500
_n_drifts = 10
_n_informative_fraction = 0.3

_chunk_size = [100, 200, 400]
_n_features = [10, 20, 40, 80]
_concept_sigmoid_spacing = [999]

_alpha = [.02, .03, .05, .07, 0.1]
_mps_ths = [0.7, 0.8, 0.85, 0.9]

_architectures = [(10), (10,10), (10,10,10), (100), (100, 100), (100, 100, 100)]

reps = 10
random_states = np.random.randint(100,10000,reps)

# res_dets = np.zeros((reps,len(_chunk_size),
#                      len(_n_features),len(_concept_sigmoid_spacing), len(_architectures),
#                      len(_alpha), len(_mps_ths),_n_chunks))

res = np.load('res/exp0_cdet_psk.npy')[:,:,:,0]
print(res.shape) # reps, chunk_size, features, architectures, alpha, mps, chunks



for ch_s_id, ch_s in enumerate(_chunk_size):
    for n_f_id, n_f in enumerate(_n_features):

            fig, ax = plt.subplots(6, 5, figsize=(20,10), sharex=True, sharey=True)
            plt.suptitle('chunk_size: %i | features: %i' % (ch_s, n_f))
            
            for arch_id, arch in enumerate(_architectures):
                for a_id, a in enumerate(_alpha):
                    
                    if arch_id==0:
                        ax[arch_id, a_id].set_title('alpha: %.3f' % (a))
                    if a_id==0:
                        ax[arch_id, a_id].set_ylabel('arch: %i' % (arch_id))

                    
                    temp = res[:, ch_s_id, n_f_id, arch_id, a_id]
                    temp = temp.swapaxes(0,1).reshape(-1,500)
                    # temp = temp.reshape(-1,500)
                    print(temp.shape)
                    
                    ax[arch_id, a_id].imshow(temp, cmap='binary', aspect='auto')
                    ax[arch_id, a_id].set_yticks([5,15,25,35], _mps_ths)
                    
            plt.tight_layout()
            plt.savefig('foo.png')
            plt.savefig('fig_psk/e0_psk_ch%i_f%i.png' % (ch_s,n_f))
            # exit()
            
            