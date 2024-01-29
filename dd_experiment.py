import numpy as np
from sklearn.neural_network import MLPClassifier
from strlearn.streams import StreamGenerator
from methods import certaintyDD
from tqdm import tqdm

_n_chunks = [200, 400]
_chunk_size = [100, 200, 400]
_n_drifts = [3,5,10]
_n_features = [10,20,30]
_n_classes = [2,5,10]
_concept_sigmoid_spacing = [5,999]

_alpha = [.02, .03, .05, .07]
epochs = 100

reps = 10
random_states = np.random.randint(100,10000,reps)

pbar = tqdm(total = reps*len(_n_chunks)
            *len(_chunk_size)*len(_n_drifts)*len(_n_features)
            *len(_n_classes)*len(_concept_sigmoid_spacing))


for n_ch_id, n_ch in enumerate(_n_chunks):
    res_dets = np.zeros((reps,len(_chunk_size),len(_n_drifts),len(_n_features),len(_n_classes),len(_concept_sigmoid_spacing),len(_alpha),n_ch))

    for r_id, rs in enumerate(random_states):
        clf = MLPClassifier(hidden_layer_sizes=(10,10,10), random_state=rs)

        for ch_s_id, ch_s in enumerate(_chunk_size):
            for n_d_id, n_d in enumerate(_n_drifts):
                for n_f_id, n_f in enumerate(_n_features):
                    for n_cl_id, n_cl in enumerate(_n_classes):
                        for conc_ss_id, conc_ss in enumerate(_concept_sigmoid_spacing):
                            
                            stream = StreamGenerator(
                                    n_chunks=n_ch,
                                    chunk_size=ch_s,
                                    n_drifts=n_d,
                                    n_classes=n_cl,
                                    n_features=n_f,
                                    n_informative=n_f,
                                    n_redundant=0,
                                    n_repeated=0, 
                                    concept_sigmoid_spacing=conc_ss)

                            dets = []
                            for a_id, a in enumerate(_alpha):
                                dets.append(certaintyDD(base_clf=clf, alpha=a, epochs=epochs))

                            for chunk_id in range(n_ch):
                                X, y = stream.get_chunk()
                                
                                # Just train
                                for det_id, det in enumerate(dets):
                                    det.partial_fit(X, y, np.arange(n_cl))
                                    if det.is_drift:
                                        res_dets[r_id,ch_s_id, n_d_id, n_f_id, n_cl_id, conc_ss_id, det_id, chunk_id] = 1
                            
                            print(res_dets[r_id,ch_s_id, n_d_id, n_f_id, n_cl_id, conc_ss_id])
                            pbar.update(1)
    
    np.save('res/dets_chunks_%i.npy' % n_ch, res_dets)
                                        
                                        
                                    