import numpy as np

def dderror(drifts_idx, detections_idx):

    if len(detections_idx) == 0: # no detections
        return np.inf, np.inf, np.inf


    n_detections = len(detections_idx)
    n_drifts = len(drifts_idx)

    ddm = np.abs(drifts_idx[:, np.newaxis] - detections_idx[np.newaxis,:])

    cdri = np.min(ddm, axis=0)
    cdec = np.min(ddm, axis=1)

    d1metric = np.mean(cdri)
    d2metric = np.mean(cdec)
    cmetric = np.abs((n_drifts/n_detections)-1)
    
    return d1metric, d2metric, cmetric
    # d1 - detection from nearest drift
    # d2 - drift from nearest detection

def find_real_drift(chunks, drifts):
    interval = round(chunks/drifts)
    idx = [interval*(i+.5) for i in range(drifts)]
    return np.array(idx).astype(int)


def indexes_to_mask(idx, n_chunks):
    arr = np.zeros(n_chunks)
    for i in idx:
        arr[i]=1
    return arr


def mask_to_indexes(mask):
    return np.argwhere(np.array(mask)==1).flatten()