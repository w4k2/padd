import numpy as np
from strlearn.streams import SemiSyntheticStreamGenerator
from tqdm import tqdm
from methods import CDET
import os

_n_chunks = 250
_chunk_size = 200
_n_drifts = 10

_n_features = [30, 60, 90]
 
_alpha = np.linspace(0.000001, 0.05, 15)
_th = np.linspace(0.000005, 0.1, 10)

_ensemble_size = 12
_replications = 12
_stat_proba = 50
_neck_width = 10

reps = 5
random_states = np.random.randint(100,10000,reps)

dataset_names = os.listdir('static_data')
try: 
    dataset_names.remove('.DS_Store')
except:
    print(dataset_names)

pbar = tqdm(total = reps*len(dataset_names)*len(_n_features)*_n_chunks)
res_dets = np.zeros((reps, len(dataset_names), len(_n_features), len(_alpha), len(_th), _n_chunks))

for data_id, data_name in enumerate(dataset_names):
    data = np.loadtxt('static_data/%s' % data_name, delimiter=',')
    X, y = data[:,:-1], data[:,-1]
    print(X.shape, y.shape)

    for rs_id, rs in enumerate(random_states):
        for n_f_id, n_f in enumerate(_n_features):
                        
            stream = SemiSyntheticStreamGenerator(
                    X, y,
                    n_chunks=_n_chunks,
                    chunk_size=_chunk_size,
                    n_drifts=_n_drifts,
                    n_features=n_f,
                    interpolation='nearest',
                    random_state=rs)

            detectors = []
            for a_id, a in enumerate(_alpha):            
                for th_id, th in enumerate(_th):     
                    detectors.append(CDET(alpha=a,  ensemble_size=_ensemble_size, n_replications=_replications,
                                            stat_proba=_stat_proba, neck_width=_neck_width, th=th))

            for chunk_id in range(_n_chunks):
                X, y = stream.get_chunk()
                
                for d_id, d in enumerate(detectors):
                    d.process(X)
                    
                    a_id = int(d_id/len(_th))
                    th_id = d_id%len(_th)
                    # print(a_id, th_id)
                    if d._is_drift:
                        res_dets[rs_id, data_id, n_f_id, a_id, th_id, chunk_id] = 1
                
            
                pbar.update(1)
                                        
                print(np.sum(res_dets[rs_id, data_id, n_f_id], axis=2))
            # exit()
            np.save('res/exp0_semi.npy', res_dets)
                                            
                                                    
                                                