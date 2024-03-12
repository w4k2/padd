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
 
_alpha = np.linspace(0.001, 0.05, 10)
_ensemble_sizes = [5,20,30]
_replications = [5,15,25,35]
_stat_probas = [75,200]
_neck_width = [256, 512]


reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_ensemble_sizes)*len(_replications)*len(_stat_probas)*len(_neck_width))

res_dets = np.zeros((reps, len(_ensemble_sizes), len(_replications), len(_stat_probas), len(_neck_width), len(_alpha), _n_chunks))

for rs_id, rs in enumerate(random_states):
    for es_id, es in enumerate(_ensemble_sizes):
        for r_id, r in enumerate(_replications):
            for sp_id, sp in enumerate(_stat_probas):
                for n_id, n in enumerate(_neck_width):
                    
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

                    detectors = []
                    for a_id, a in enumerate(_alpha):               
                        detectors.append(CDET(alpha=a,  ensemble_size=es, n_replications=r,
                                              stat_proba=sp, neck_width=n))
                                    
                    for chunk_id in range(_n_chunks):
                        X, y = stream.get_chunk()
                        
                        for d_id, d in enumerate(detectors):
                            d.process(X)
                            if d.is_drift:
                                res_dets[rs_id, es_id, r_id, sp_id, n_id, d_id, chunk_id] = 1
                        
                    
                    pbar.update(1)
                        
                    print(np.sum(res_dets[rs_id, es_id, r_id, sp_id, n_id], axis=1))

                np.save('res/mini_exp.npy', res_dets)
                                            
                                                    
                                                