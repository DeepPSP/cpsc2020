"""
"""
import os
import math
from typing import Union, Optional, NoReturn
from numbers import Real

import numpy as np
from scipy.signal import resample_poly
try:
    import biosppy.signals.ecg as BSE
except:
    import references.biosppy.biosppy.signals.ecg as BSE

from .ecg_rpeaks_dl_models import load_model
from utils import mask_to_intervals


__all__ = [
    "seq_lab_net_detect",
]


def seq_lab_net_detect(sig:np.ndarray, fs:Real, **kwargs) -> np.ndarray:
    """ finished, NOT checked,

    model of entry 0416 of CPSC2019

    Parameters:
    -----------
    sig: ndarray,
        the (raw) ECG signal
    fs: real number,
        sampling frequency of `sig`
    kwargs: dict,
        not used, to keep in accordance with other rpeak detection function

    Returns:
    --------
    rpeaks: ndarray,
        indices of rpeaks in `sig`

    References:
    -----------
    [1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
    """
    verbose = kwargs.get("verbose", 0)

    cnn_model, crnn_model = load_model("ecg_seq_lab_net")
    model_fs = 500
    model_input_len = 5000

    half_overlap_len = 500
    overlap_len = 2 * half_overlap_len
    forward_len = model_input_len - overlap_len
    batch_size = 128

    # pre-process
    sig_rsmp = _remove_spikes_naive(sig)
    # TODO: consider "To achieve better model generalization, 
    # the mean of signal values is subtracted" in ref. [1]

    if fs != model_fs:
        sig_rsmp = resample_poly(sig_rsmp, up=model_fs, down=int(fs))
    else:
        sig_rsmp = np.array(sig_rsmp).copy()

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

    # prob --> qrs mask --> qrs intervals --> rpeaks
    rpeaks = _seq_lab_net_post_process(prob, 0.5)

    # convert from resampled positions to original positions
    rpeaks = (np.round((fs/model_fs) * rpeaks)).astype(int)
    rpeaks = rpeaks[np.where(rpeaks < len(sig))[0]]

    # adjust to the "true" rpeaks, 
    # i.e. the max in a small nbh of each element in `rpeaks`
    rpeaks = BSE.correct_rpeaks(
        signal=sig,
        rpeaks=rpeaks,
        sampling_rate=fs,
        tol=0.05,
    )
    return rpeaks


def _seq_lab_net_post_process(prob:np.ndarray, prob_thr:float=0.5) -> np.ndarray:
    """ finished, checked,

    convert the array of probability predictions into the array of indices of rpeaks

    Parameters:
    -----------
    prob: ndarray,
    prob_thr: float, default 0.5,

    Returns:
    --------
    rpeaks: ndarray,
        indices of rpeaks in converted from the array `prob`
    """
    _prob = prob.squeeze()
    assert _prob.ndim == 1, \
        "only support single record processing, batch processing not supported!"
    # prob --> qrs mask --> qrs intervals --> rpeaks
    mask = (_prob > prob_thr).astype(int)
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
                if _prob[int(rpeaks[r]/8)] > _prob[int(rpeaks[r+1]/8)]:
                    rpeaks = np.delete(rpeaks, r+1)
                    check = True
                    break
                else:
                    rpeaks = np.delete(rpeaks, r)
                    check = True
                    break
            check = False
    return rpeaks


def _remove_spikes_naive(sig:np.ndarray) -> np.ndarray:
    """ finished, NOT checked,

    remove `spikes` from `sig` using a naive method proposed in entry 0416 of CPSC2019

    `spikes` here refers to abrupt large bumps with (abs) value larger than 20 mV,
    do NOT confuse with `spikes` in paced rhythm

    Parameters:
    -----------
    sig: ndarray,
        single-lead ECG signal with potential spikes
    
    Returns:
    --------
    filtered_sig: ndarray,
        ECG signal with `spikes` removed
    """
    b = list(filter(lambda k: k > 0, np.argwhere(np.abs(sig)>20).squeeze())
    filtered_sig = sig.copy()
    for k in b:
        filtered_sig[k] = filtered_sig[k-1]
    return filtered_sig
