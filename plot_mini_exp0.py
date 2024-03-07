import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


_n_chunks = 500
_alpha = np.linspace(0.001, 0.01, 10)


reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_alpha))


res_dets = np.load('res/mini_exp.npy')

print(res_dets.shape)                    
                                    
                                    
fig, ax = plt.subplots(2,2,figsize=(15,10), sharex=True, sharey=True)
ax = ax.ravel()

m_labels = ['cdet false', 'psk false', 'cdet true', 'psk true']
for m in range(4):

    aa = res_dets[:,:,m].swapaxes(0,1).reshape(-1,_n_chunks)
    ax[m].imshow(aa, cmap='binary', aspect='auto')
    ax[m].set_title(m_labels[m])

for aa in ax:
    aa.set_yticks(np.linspace(5,95,10), ['%.5f' % a for a in _alpha])
plt.tight_layout()
plt.savefig('foo.png')