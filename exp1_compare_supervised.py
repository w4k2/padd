import numpy as np
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from detectors.meta import MetaClassifier
from detectors.adwin import ADWIN
from detectors.ddm import DDM
from detectors.eddm import EDDM
from sklearn.naive_bayes import GaussianNB

np.random.seed(1410)

_n_chunks = 250
_chunk_size = 200
_n_drifts = 10
_n_informative_fraction = 0.3

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]

alphas = [0.13, 0.07] # for css 5, 999
ths = [0.26, 0.19] # for css 5, 999

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_n_features)*len(_concept_sigmoid_spacing)*_n_chunks)
res_dets = np.zeros((reps, len(_n_features), len(_concept_sigmoid_spacing), 3, _n_chunks))

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
                MetaClassifier(base_clf=GaussianNB(), detector=ADWIN()),
                MetaClassifier(base_clf=GaussianNB(), detector=DDM()),
                MetaClassifier(base_clf=GaussianNB(), detector=EDDM())
            ]      
            
            
            for chunk_id in range(_n_chunks):
                X, y = stream.get_chunk()
            
                
                for d_id, d in enumerate(dets):
                    if chunk_id==0:
                        d.partial_fit(X, y, np.unique(y))
                    else:
                        d.predict(X)
                        d.partial_fit(X, y, np.unique(y))
                        
                        if d.detector.drift[-1]==2:
                            res_dets[rs_id, n_f_id, css_id, d_id, chunk_id] = 1

                pbar.update(1)
            print(rs_id, n_f_id, css_id, d_id)
            print(np.sum(res_dets[rs_id, n_f_id, css_id], axis=1))

            np.save('res/exp1_comp_v2_sup.npy', res_dets)
                                    
                                    
                                