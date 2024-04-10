import numpy as np
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from methods import PADD

_n_chunks = 250
_chunk_size = 200
_n_drifts = 10
_n_informative_fraction = 0.3

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
 
_alpha = np.linspace(0.03, 0.2, 15)
_th = np.linspace(0.1, 0.3, 10)

_ensemble_size = 12
_replications = 12
_stat_proba = 50
_neck_width = 10

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_n_features)*len(_concept_sigmoid_spacing)*_n_chunks)
res_dets = np.zeros((reps, len(_n_features), len(_concept_sigmoid_spacing), len(_alpha), len(_th), _n_chunks))

for rs_id, rs in enumerate(random_states):
    for n_f_id, n_f in enumerate(_n_features):
        for css_id, css in enumerate(_concept_sigmoid_spacing):
                    
            stream = StreamGenerator(
                    n_chunks=_n_chunks,
                    chunk_size=_chunk_size,
                    n_drifts=_n_drifts,
                    n_classes=2,
                    n_features=n_f,
                    n_informative=int(_n_informative_fraction * n_f),
                    n_redundant=0,
                    n_repeated=0, 
                    concept_sigmoid_spacing=css,
                    random_state=rs)

            detectors = []
            for a_id, a in enumerate(_alpha):            
                for th_id, th in enumerate(_th):     
                    detectors.append(PADD(alpha=a,  ensemble_size=_ensemble_size, n_replications=_replications,
                                            stat_proba=_stat_proba, neck_width=_neck_width, th=th))

            for chunk_id in range(_n_chunks):
                X, y = stream.get_chunk()
                
                for d_id, d in enumerate(detectors):
                    d.process(X)
                    
                    a_id = int(d_id/len(_th))
                    th_id = d_id%len(_th)

                    if d._is_drift:
                        res_dets[rs_id, n_f_id, css_id, a_id, th_id, chunk_id] = 1
                
            
                pbar.update(1)
                                        
                print(np.sum(res_dets[rs_id, n_f_id, css_id], axis=2))

            np.save('res/mini_exp_v2.npy', res_dets)
                                            
                                                    
                                                