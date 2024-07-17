import numpy as np
import matplotlib.pyplot as plt
from strlearn.streams import StreamGenerator


def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

_n_chunks = 250

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
_n_drifts = [3, 5, 10, 15]
 
alphas = [0.13, 0.07] # for css 5, 999
ths = [0.26, 0.19] # for css 5, 999

reps = 10
n_methods = 4
cps = np.linspace(0, 250, 200*250)

res_dets = np.load('res/exp1_comp_final.npy') 
print(res_dets.shape)
# replications, features, concept sigmoid, n_drifts, detectors, chunks

# res_dets = res_dets[:,:,:,:,[0,1,3,4,5,6]]
det_names = ['MD3', 'OCDD', 'CDDD', 'PADD', 'ADWIN', 'DDM', 'EDDM']

conceptsss = np.zeros((2, len(_n_drifts), 250*200))

for css_id, css in enumerate(_concept_sigmoid_spacing):
    for d_id, d in enumerate(_n_drifts):
        
        sc = {'n_features':1, 'n_informative':1, 
              'n_clusters_per_class':1, 'n_redundant':0,
              'concept_sigmoid_spacing':css,
              'n_drifts':d, 'n_chunks':_n_chunks, 'chunk_size':200}
        s = StreamGenerator(**sc)
        s._make_classification()
        conceptsss[css_id,d_id] = (s.concept_probabilities)

for css_id, css in enumerate(_concept_sigmoid_spacing):
    fig, ax = plt.subplots(len(_n_drifts), len(_n_features), figsize=(12, 13), sharey=True)
    # fig.suptitle("%s" % ('sudden' if css==999 else 'gradual'), fontsize=12)

    for f_id, f in enumerate(_n_features):
        for d_id, d in enumerate(_n_drifts):
            
            aa = ax[d_id, f_id]
            r = res_dets[:,f_id, css_id, d_id]
            print(r.shape)
            for det in range(len(det_names)):
                print(det, np.unique(r[:,det], return_counts=True))
                for rep in range(10):
                    step = 1
                    start = det*10 + step*rep
                    stop = det*10 + step*(rep+1)
                    detections = np.argwhere(r[rep,det]==1).flatten()
                    
                    aa.vlines(detections, start, stop, color='black' if det != 3 else 'red')
                    
            aa.plot(cps, conceptsss[css_id,d_id]*5-7.5, c='red')
            aa.grid(ls=":")
        
            if d_id==0:          
                aa.set_title('%i dim' % (f))
            if f_id==0:
                aa.set_ylabel('%s drifts' % (d), fontsize=12)
            
            drfs = get_real_drfs(_n_chunks, d)
            
            aa.set_xticks(drfs, ['D%02d' % i for i in range(len(drfs))], rotation=90)
            aa.set_yticks([(10*i)-5 
                        for i in range(len(det_names)+1)], 
                        ['concept']+det_names)
            aa.spines['top'].set_visible(False)
            aa.spines['right'].set_visible(False)
            aa.spines['bottom'].set_visible(False)
            
        plt.tight_layout()
        plt.savefig('foo.png')
        plt.savefig("fig/e2_%i.png" % (css))
        plt.savefig("fig/e2_%i.eps" % (css))
        # exit()