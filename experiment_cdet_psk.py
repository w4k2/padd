import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from methods import CDET_PSK

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

pbar = tqdm(total = reps
            *len(_chunk_size)*len(_n_features)
            *len(_concept_sigmoid_spacing)*len(_architectures)
            *len(_alpha)*len(_mps_ths))


res_dets = np.zeros((reps,len(_chunk_size),
                     len(_n_features),len(_concept_sigmoid_spacing), len(_architectures),
                     len(_alpha), len(_mps_ths),_n_chunks))


for r_id, rs in enumerate(random_states):
        for ch_s_id, ch_s in enumerate(_chunk_size):
            for n_f_id, n_f in enumerate(_n_features):
                for conc_ss_id, conc_ss in enumerate(_concept_sigmoid_spacing):

                    for arch_id, arch in enumerate(_architectures):
                        for a_id, a in enumerate(_alpha):
                            for mps_id, mps in enumerate(_mps_ths):
        
                                stream = StreamGenerator(
                                        n_chunks=_n_chunks,
                                        chunk_size=ch_s,
                                        n_drifts=_n_drifts,
                                        n_classes=2,
                                        n_features=n_f,
                                        n_informative=int(_n_informative_fraction * n_f),
                                        n_redundant=0,
                                        n_repeated=0, 
                                        concept_sigmoid_spacing=conc_ss,
                                        random_state=rs)

                                                            
                                base = MLPClassifier(hidden_layer_sizes=arch, random_state=rs)
                                det = CDET_PSK(base_mlp=base, alpha=a, mps_th=mps)
                                
                                for chunk_id in range(_n_chunks):
                                    X, y = stream.get_chunk()
                                    
                                    det.process(X, y)
                                    if det.is_drift:
                                        res_dets[r_id, ch_s_id, n_f_id, conc_ss_id, arch_id, a_id, mps_id, chunk_id] = 1
                            
                                    # exit()
                                pbar.update(1)
                            print(np.sum(res_dets[r_id, ch_s_id, n_f_id, conc_ss_id, arch_id, a_id], axis=1))

                        np.save('res/exp0_cdet_psk.npy', res_dets)
                                                    
                                        
                                    