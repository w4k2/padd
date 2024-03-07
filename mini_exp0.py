import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from methods import CDET

_n_chunks = 500
_n_drifts = 10
_n_informative_fraction = 0.3
_chunk_size = 200
_n_features = 30
_concept_sigmoid_spacing = 999
_architecture = (10,10,10)
_mps_th = 0.9
 
# _alpha = np.linspace(1e-12, 1e-6, 10)
_alpha = np.linspace(0.001, 0.01, 10)

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_alpha))

n_detectors = 4
res_dets = np.zeros((reps,len(_alpha),n_detectors,_n_chunks))


for r_id, rs in enumerate(random_states):
    for a_id, a in enumerate(_alpha):

        stream = StreamGenerator(
                n_chunks=_n_chunks,
                chunk_size=_chunk_size,
                n_drifts=_n_drifts,
                n_classes=2,
                n_features=_n_features,
                n_informative=int(_n_informative_fraction * _n_features),
                n_redundant=0,
                n_repeated=0, 
                concept_sigmoid_spacing=_concept_sigmoid_spacing,
                random_state=rs)

                                    
        base = MLPClassifier(hidden_layer_sizes=_architecture, random_state=rs)
        
        detectors = [
            CDET(base_mlp=base, alpha=a, n_epochs=50, mps_th=None, balanced=False, psk=False),
            CDET(base_mlp=base, alpha=a, n_epochs=None, mps_th=_mps_th, balanced=False, psk=True),
            CDET(base_mlp=base, alpha=a, n_epochs=50, mps_th=None, balanced=True, psk=False),
            CDET(base_mlp=base, alpha=a, n_epochs=None, mps_th=_mps_th, balanced=True, psk=True),
        ]
        
        for chunk_id in range(_n_chunks):
            X, y = stream.get_chunk()
            
            for d_id, d in enumerate(detectors):
                d.process(X, y)
                if d.is_drift:
                    res_dets[r_id, a_id, d_id, chunk_id] = 1
            
        
        pbar.update(1)
            
        print(np.sum(res_dets[r_id, a_id], axis=1))

    np.save('res/mini_exp.npy', res_dets)
                                
                                        
                                    