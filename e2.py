import numpy as np
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import StreamGenerator
from tqdm import tqdm
from detectors.adwin import ADWIN
from detectors.ddm import DDM
from detectors.eddm import EDDM
from detectors.meta import MetaClassifier
from methods import PADD
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import OneClassDriftDetector
from reference.MD3 import MD3

np.random.seed(1410)

_n_chunks = 250
_chunk_size = 200
_n_informative_fraction = 0.3

_n_drifts = [3, 5, 10, 15]
_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
 
alphas = [0.13, 0.07] # for css 5, 999
ths = [0.26, 0.19] # for css 5, 999

oc_percentage = [0.75, 0.9, 0.999] # for 30. 60, 90 features
md3_sigma = [0.15, 0.1, 0.08] # for 30. 60, 90 features

cd_sensitivity = [0.2, 0.2, 0.9, 0.9] # for 3, 5, 10, 15 drifts

_ensemble_size = 12
_replications = 12
_stat_proba = 50
_neck_width = 10

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_n_features)*len(_concept_sigmoid_spacing)*len(_n_drifts)*_n_chunks)
res_dets = np.zeros((reps, len(_n_features), len(_concept_sigmoid_spacing), len(_n_drifts), 7, _n_chunks))

for rs_id, rs in enumerate(random_states):
    for n_f_id, n_f in enumerate(_n_features):
        for css_id, css in enumerate(_concept_sigmoid_spacing):
            for n_d_id, n_d in enumerate(_n_drifts):
                stream = StreamGenerator(
                        n_chunks=_n_chunks,
                        chunk_size=_chunk_size,
                        n_drifts=n_d,
                        n_classes=2,
                        n_features=n_f,
                        n_informative=int(_n_informative_fraction * n_f),
                        n_redundant=0,
                        n_repeated=0, 
                        concept_sigmoid_spacing=css,
                        random_state=rs)

                dets = [
                    MD3(sigma=md3_sigma[n_f_id]),
                    OneClassDriftDetector(size = 250, dim = n_f, percent = oc_percentage[n_f_id], nu=0.5),
                    CentroidDistanceDriftDetector(sensitive = cd_sensitivity[n_d_id]),
                    PADD(alpha=alphas[css_id], ensemble_size=_ensemble_size, n_replications=_replications,
                                                stat_proba=_stat_proba, neck_width=_neck_width, th=ths[css_id]),
                    MetaClassifier(base_clf=GaussianNB(), detector=ADWIN()),
                    MetaClassifier(base_clf=GaussianNB(), detector=DDM()),
                    MetaClassifier(base_clf=GaussianNB(), detector=EDDM())
                ]
                
                
                for chunk_id in range(_n_chunks):
                    X, y = stream.get_chunk()
                    
                    for d_id, d in enumerate(dets):
                        if d_id==0:
                            d.process(X, y)
                        elif d_id in [1,2,3]: 
                            d.process(X)
                        else:
                            # test than train for supervised
                            if chunk_id==0:
                                d.partial_fit(X, y, np.unique(y))
                            else:
                                d.predict(X)
                                d.partial_fit(X, y, np.unique(y))
                                
                            
                        if d_id<4:
                            #unsupervised
                            if d._is_drift:
                                res_dets[rs_id, n_f_id, css_id, n_d_id, d_id, chunk_id] = 1
                        else:
                            #supervised
                            if chunk_id>0:
                                if d.detector.drift[-1]==2:
                                    res_dets[rs_id, n_f_id, css_id, n_d_id, d_id, chunk_id] = 1


                    pbar.update(1)
                print(rs_id, n_f_id, css_id, n_d_id, d_id)
                print(np.sum(res_dets[rs_id, n_f_id, css_id, n_d_id], axis=1))

                np.save('res/exp1_comp_final.npy', res_dets)
                                        
                                        
                                    