import numpy as np
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from methods import CDET
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import OneClassDriftDetector
from reference.MD3 import MD3

np.random.seed(12221)

_n_chunks = 250
_n_informative_fraction = 0.3

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
res_dets = np.zeros((reps,len(_n_drifts),len(_chunk_size),
                     len(_n_features),len(_concept_sigmoid_spacing),
                     len(_n_classes),n_methods,_n_chunks))


for r_id, rs in enumerate(random_states):
    for n_d_id, n_d in enumerate(_n_drifts):
        for ch_s_id, ch_s in enumerate(_chunk_size):
            for n_f_id, n_f in enumerate(_n_features):
                for conc_ss_id, conc_ss in enumerate(_concept_sigmoid_spacing):
                    for n_cl_id, n_cl in enumerate(_n_classes):

                        stream = StreamGenerator(
                                n_chunks=_n_chunks,
                                chunk_size=ch_s,
                                n_drifts=n_d,
                                n_classes=n_cl,
                                n_features=n_f,
                                n_informative=int(_n_informative_fraction * n_f),
                                n_redundant=0,
                                n_repeated=0, 
                                concept_sigmoid_spacing=conc_ss,
                                random_state=rs)

                                                        
                        dets = [
                            MD3(sigma=0.15),
                            OneClassDriftDetector(size = ch_s, dim = n_f, percent = 0.85, nu=0.5),
                            CentroidDistanceDriftDetector(sensitive=0.2),
                            CDET(alpha=0.006, ensemble_size=35, n_replications=20, stat_proba=200, neck_width=512),
                            CDET(alpha=0.012, ensemble_size=35, n_replications=20, stat_proba=200, neck_width=512),
                            CDET(alpha=0.024, ensemble_size=35, n_replications=20, stat_proba=200, neck_width=512),
                            ]
                        
                        for chunk_id in range(_n_chunks):
                            X, y = stream.get_chunk()
                            
                            for d_id, d in enumerate(dets):
                                if d_id==0:
                                    d.process(X, y)
                                else: 
                                    d.process(X)
                                    
                                if d._is_drift:
                                    res_dets[r_id, n_d_id, ch_s_id, n_f_id, conc_ss_id, n_cl_id, d_id, chunk_id] = 1

                            pbar.update(1)
                        print(r_id, n_d, ch_s, n_f, conc_ss, n_cl)
                        print(np.sum(res_dets[r_id, n_d_id, ch_s_id, n_f_id, conc_ss_id, n_cl_id], axis=1))

                        np.save('res/exp1_comp.npy', res_dets)
                                                
                                    
                                