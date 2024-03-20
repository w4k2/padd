import numpy as np
from sklearn.naive_bayes import GaussianNB
from strlearn.streams import SemiSyntheticStreamGenerator
from tqdm import tqdm
from detectors.adwin import ADWIN
from detectors.ddm import DDM
from detectors.eddm import EDDM
from detectors.meta import MetaClassifier
from methods import CDET
from reference.CDDD import CentroidDistanceDriftDetector
from reference.OCDD import OneClassDriftDetector
from reference.MD3 import MD3
import os

np.random.seed(1410)

_n_chunks = 250
_chunk_size = 200

_n_drifts = [3, 5, 10, 15]
_n_features = [30, 60, 90]
_concept_sigmoid_spacing = ['cubic','nearest']
 
alphas = [0.13, 0.07] # for css 5, 999
ths = [0.26, 0.19] # for css 5, 999

_ensemble_size = 12
_replications = 12
_stat_proba = 50
_neck_width = 10

reps = 5
random_states = np.random.randint(100,10000,reps)

dataset_names = os.listdir('static_data')

pbar = tqdm(total = reps*len(dataset_names)*len(_n_features)*len(_concept_sigmoid_spacing)*len(_n_drifts)*_n_chunks)
res_dets = np.zeros((reps, len(dataset_names), len(_n_features), len(_concept_sigmoid_spacing), len(_n_drifts), 7, _n_chunks))

for data_id, data_name in enumerate(dataset_names):
    data = np.loadtxt('static_data/%s' % data_name, delimiter=',')
    X, y = data[:,:-1], data[:,-1]
    print(X.shape, y.shape)

    for rs_id, rs in enumerate(random_states):
        for n_f_id, n_f in enumerate(_n_features):
            for css_id, css in enumerate(_concept_sigmoid_spacing):
                for n_d_id, n_d in enumerate(_n_drifts):
                    stream = SemiSyntheticStreamGenerator(
                        X, y,
                        n_chunks=_n_chunks,
                        chunk_size=_chunk_size,
                        n_drifts=n_d,
                        n_features=n_f,
                        interpolation=css,
                        random_state=rs)

                    dets = [
                        MD3(sigma=0.1),
                        OneClassDriftDetector(size = 250, dim = n_f, percent = 0.9, nu=0.5),
                        CentroidDistanceDriftDetector(sensitive = 0.2),
                        CDET(alpha=alphas[css_id], ensemble_size=_ensemble_size, n_replications=_replications,
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
                                    res_dets[rs_id, data_id, n_f_id, css_id, n_d_id, d_id, chunk_id] = 1
                            else:
                                #supervised
                                if chunk_id>0:
                                    if d.detector.drift[-1]==2:
                                        res_dets[rs_id, data_id, n_f_id, css_id, n_d_id, d_id, chunk_id] = 1


                        pbar.update(1)
                    print(rs_id, data_id, n_f_id, css_id, n_d_id, d_id)
                    print(np.sum(res_dets[rs_id, data_id, n_f_id, css_id, n_d_id], axis=1))

                    np.save('res/exp1_semisyn.npy', res_dets)
                                            
                                            
                                        