import numpy as np
from tabulate import tabulate
from scipy.stats import ttest_rel

def dderror(drifts, detections, n_chunks):

    if len(detections) == 0: # no detections
        detections = np.arange(n_chunks)

    n_detections = len(detections)
    n_drifts = len(drifts)

    ddm = np.abs(drifts[:, np.newaxis] - detections[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    d1metric = np.mean(cdri)
    d2metric = np.mean(cdec)
    cmetric = np.abs((n_drifts/n_detections)-1)

    return d1metric, d2metric, cmetric
    # d1 - detection from nearest drift
    # d2 - drift from nearest detection

def get_real_drfs(n_chunks, n_drifts):
    real_drifts = np.linspace(0,n_chunks,n_drifts+1)[:-1]
    real_drifts += (real_drifts[1]/2)
    return real_drifts

_n_chunks = 250

_n_features = [30, 60, 90]
_concept_sigmoid_spacing = [5, 999]
_n_drifts = [3, 5, 10, 15]

res_dets = np.load('res/exp1_comp_final.npy') 
print(res_dets.shape)
# replications, features, concept sigmoid, n_drifts, detectors, chunks

res_dets = res_dets[:,:,:,:,[0,1,3,4,5,6]]
det_names = ['MD3', 'OCDD', 'PADD', 'ADWIN', 'DDM', 'EDDM']

results_all = np.zeros((10, len(_n_features), len(_concept_sigmoid_spacing), len(_n_drifts), len(det_names), 3))

metric_names = ["d1", "d2", "cnt_ratio"]

#features, drifts
for f_id, f in enumerate(_n_features):
    for d_id, d in enumerate(_n_drifts):
        for css_id, css in enumerate(_concept_sigmoid_spacing):

                dderror_arr = np.zeros((10, len(det_names), 3))

                for rep in range(10):
                    for det in range(len(det_names)):
                        det_drf = np.argwhere(res_dets[rep, f_id, css_id, d_id, det] > 0).flatten()
                        real_drf = get_real_drfs(n_chunks=_n_chunks, n_drifts=d).astype(int)
                        print(real_drf)
                        print(det_drf)
                
                        err = dderror(real_drf, det_drf, _n_chunks)
                        dderror_arr[rep, det] = err

                results_all[:, f_id, css_id, d_id] = dderror_arr
                
                
results_all_mean = np.mean(results_all, axis=0)
print(results_all_mean.shape)
# features, concept sigmoid, n_drifts, detectors, metrics

#for every metric
for metric_id in range(3):

    alpha = 0.05

    t = []
    t.append(["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)"])


    for css_id, css in enumerate(_concept_sigmoid_spacing):
        for n_d_id, n_d in enumerate(_n_drifts):
            for n_f_id, n_f in enumerate(_n_features):
                
                temp = results_all[:,n_f_id,css_id,n_d_id]
                print(temp.shape)
                str_name = '%s %iD %iF' % ('S' if css==999 else 'G', n_d, n_f)                
                
                """
                t-test
                """

                metric_temp = temp[:, :, metric_id]
                length = 6

                s = np.zeros((length, length))
                p = np.zeros((length, length))

                for i in range(length):
                    for j in range(length):
                        s[i, j], p[i, j] = ttest_rel(metric_temp.T[i], metric_temp.T[j])

                _ = np.where((p < alpha) * (s < 0))
                conclusions = [list(1 + _[1][_[0] == i]) for i in range(length)]

                t.append(["%s" % str_name] + ["%.3f" % v for v in results_all_mean[n_f_id,css_id,n_d_id, :, metric_id]])
                t.append([''] + [", ".join(["%i" % i for i in c])
                                if len(c) > 0 and len(c) < length-1 else ("all" if len(c) == length-1 else "---")
                                for c in conclusions])


    with open('tables/table_%s.txt' % (metric_names[metric_id]), 'w') as f:
        f.write(tabulate(t, det_names,floatfmt="%.3f", tablefmt="latex_booktabs"))
        
    # exit()
