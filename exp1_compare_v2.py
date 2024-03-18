import numpy as np
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from methods import CDET
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import OneClassDriftDetector
from reference.MD3 import MD3

_n_chunks = 250
_chunk_size = 200
_n_drifts = 10
_n_informative_fraction = 0.3

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
 
alphas = [0.13, 0.07] # for css 5, 999
ths = [0.26, 0.19] # for css 5, 999
_ensemble_size = 12
_replications = 12
_stat_proba = 50
_neck_width = 10

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_n_features)*len(_concept_sigmoid_spacing)*_n_chunks)
res_dets = np.zeros((reps, len(_n_features), len(_concept_sigmoid_spacing), 4, _n_chunks))

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

            dets = [
                MD3(sigma=0.075),
                OneClassDriftDetector(size = 250, dim = n_f, percent = 0.8, nu=0.5),
                CentroidDistanceDriftDetector(sensitive = 0.2),
                CDET(alpha=alphas[css_id], ensemble_size=_ensemble_size, n_replications=_replications,
                                            stat_proba=_stat_proba, neck_width=_neck_width, th=ths[css_id]),
            ]       
            
            
            for chunk_id in range(_n_chunks):
                X, y = stream.get_chunk()
                
                for d_id, d in enumerate(dets):
                    if d_id==0:
                        d.process(X, y)
                    else: 
                        d.process(X)
                        
                    if d._is_drift:
                        res_dets[rs_id, n_f_id, css_id, d_id, chunk_id] = 1

                pbar.update(1)
            print(rs_id, n_f_id, css_id, d_id)
            print(np.sum(res_dets[rs_id, n_f_id, css_id], axis=1))

            np.save('res/exp1_comp_v2.npy', res_dets)
                                    
                                    
                                