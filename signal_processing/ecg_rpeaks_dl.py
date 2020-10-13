"""
"""
import os
import math

import numpy as np
from scipy.signal import resample_poly
try:
    import biosppy.signals.ecg as BSE
except:
    import references.biosppy.biosppy.signals.ecg as BSE

from .ecg_rpeaks_dl_models import load_model
from utils import mask_to_intervals


def seq_lab_net_detect(sig:np.ndarray, fs:Real, **kwargs) -> np.ndarray:
    """ finished, NOT checked,

    Parameters:
    -----------
    sig: ndarray,
        the (raw) ECG signal
    fs: real number,
        sampling frequency of `sig`

    Returns:
    --------
    rpeaks: ndarray,
        indices of rpeaks in `sig`
    """
    cnn_model, crnn_model = load_model("ecg_seq_lab_net")

    model_fs = 500
    model_input_len = 5000

    half_overlap_len = 500
    overlap_len = 2 * half_overlap_len
    forward_len = model_input_len - overlap_len
    batch_size = 128

    if fs != model_fs:
        sig_rsmp = resample_poly(sig, up=model_fs, down=int(fs))
    else:
        sig_rsmp = np.array(sig).copy()

    n_segs, residue = divmod(len(sig_rsmp), forward_len)
    if residue != 0:
        sig_rsmp = np.append(sig_rsmp, np.zeros((forward_len-residue,)))
        n_segs += 1

    n_batches = math.ceil(n_segs / batch_size)

    prob = []
    segs = list(range(n_segs))
    for b_idx in range(n_batches):
        # b_start = b_idx * batch_size * forward_len
        b_start = b_dix * batch_size
        b_segs = segs[b_start: b_start + batch_size]
        b_input = np.vstack(
            [sig_rsmp[idx*forward_len: idx*forward_len+model_input_len] for idx in b_segs]
        ).reshape((-1, model_input_len, 1))
        prob_cnn = cnn_model.predict(b_input)
        prob_crnn = crnn_model.predict(b_input)
        b_prob = (prob_cnn[...,0] + prob_crnn[...,0]) / 2
        b_prob = b_prob[..., half_overlap_len: -half_overlap_len]
        prob += b_prob.flatten()
    # prob, output from the for loop,
    # is the array of probabilities for sig_rsmp[half_overlap_len: -half_overlap_len]
    prob = list(repeat(0,half_overlap_len)) + prob + list(repeat(0,half_overlap_len))
    prob = np.array(prob)

    # prob to mask, mask to intervals, intervals to rpeaks
    mask = (prob > 0.5).astype(int)
    qrs_intervals = mask_to_intervals(mask, 1)
    # should be 8 * (itv[0]+itv[1]) / 2
    rpeaks = 4 * np.array([itv[0]+itv[1] for itv in qrs_intervals])

    # post-process
    rpeaks_diff = np.diff(rpeaks)
    check = True
    while check:
        rpeaks_diff = np.diff(rpeaks)
        for r in range(len(rpeaks_diff)):
            if rpeaks_diff[r] < 100:  # 200 ms
                if prob[int(rpeaks[r]/8)] > prob[int(rpeaks[r+1]/8)]:
                    rpeaks = np.delete(rpeaks, r+1)
                    check = True
                    break
                else:
                    rpeaks = np.delete(rpeaks, r)
                    check = True
                    break
            check = False

    # convert from resampled positions to original positions
    rpeaks = (np.round((fs/model_fs) * rpeaks)).astype(int)

    # adjust to the "true" rpeaks, 
    # i.e. the max in a small nbh of each element in `rpeaks`
    rpeaks = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=0.05,
    )

    return rpeaks
    