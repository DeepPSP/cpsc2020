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


CNN_MODEL, CRNN_MODEL = load_model("ecg_seq_lab_net")


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
    batch_size = kwargs.get("batch_size", None)

    model_fs = 500
    model_granularity = 8  # 1/8 times of model_fs

    # pre-process
    sig_rsmp = _seq_lab_net_pre_process(sig)

    if fs != model_fs:
        sig_rsmp = resample_poly(sig_rsmp, up=model_fs, down=int(fs))
    else:
        sig_rsmp = np.array(sig_rsmp).copy()

    if batch_size is not None:
        model_input_len = 5000

        half_overlap_len = 500
        overlap_len = 2 * half_overlap_len
        forward_len = model_input_len - overlap_len

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
            prob_cnn = CNN_MODEL.predict(b_input)
            prob_crnn = CRNN_MODEL.predict(b_input)
            b_prob = (prob_cnn[...,0] + prob_crnn[...,0]) / 2
            b_prob = b_prob[..., half_overlap_len: -half_overlap_len]
            prob += b_prob.flatten()
        # prob, output from the for loop,
        # is the array of probabilities for sig_rsmp[half_overlap_len: -half_overlap_len]
        prob = list(repeat(0,half_overlap_len)) + prob + list(repeat(0,half_overlap_len))
        prob = np.array(prob)
    else:
        prob_cnn = cnn_model.predict(sig_rsmp.reshape((1,len(sig_rsmp),1)))
        prob_crnn = crnn_model.predict(sig_rsmp.reshape((1,len(sig_rsmp),1)))
        prob = ((prob_cnn + prob_crnn) / 2).squeeze()

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


def _seq_lab_net_pre_process(sig:np.ndarray) -> np.ndarray:
    """ NOT finished, NOT checked,

    Parameters:
    -----------
    sig: ndarray,
        the ECG signal to be pre-processed

    Returns:
    --------
    sig_processed: ndarray,
        the processed ECG signal
    """
    # Single towering spike whose voltage is more than 20 mV is examined 
    # and replaced by the normal sample immediately before it
    sig_processed = _remove_spikes_naive(sig)
    # TODO:
    # To achieve better model generalization,
    # the (local?) mean of signal values is subtracted for each recording
    return sig_processed


def _seq_lab_net_post_process(prob:np.ndarray, prob_thr:float=0.5, duration_thr:int=4*16, dist_thr:Union[int,Sequence[int]]=[200,1200]) -> np.ndarray:
    """ finished, checked,

    convert the array of probability predictions into the array of indices of rpeaks

    Parameters:
    -----------
    prob: ndarray,
        the array of probabilities of qrs complex
    prob_thr: float, default 0.5,
        threshold of probability for predicting qrs complex
    duration_thr: int, default 4*16,
        minimum duration for a "true" qrs complex, units in ms
    dist_thr: int or sequence of int, default [200, 1200],
        0. minimum distance for two consecutive qrs complexes, units in ms;
        1. maximum distance for checking missing qrs complexes, units in ms
        if is int, then is 0.

    Returns:
    --------
    rpeaks: ndarray,
        indices of rpeaks in converted from the array `prob`
    """
    model_fs = 500
    model_granularity = 8  # 1/8 times of model_fs
    _prob = prob.squeeze()
    assert _prob.ndim == 1, \
        "only support single record processing, batch processing not supported!"
    # prob --> qrs mask --> qrs intervals --> rpeaks
    mask = (_prob >= prob_thr).astype(int)
    qrs_intervals = mask_to_intervals(mask, 1)
    # threshold of 64 ms for the duration of clustering positive samples
    # is set to eliminate some wrong predictions
    _duration_thr = duration_thr / (1000/model_fs) / model_granularity
    # should be 8 * (itv[0]+itv[1]) / 2
    rpeaks = (model_granularity//2) * np.array([itv[0]+itv[1] for itv in qrs_intervals if itv[1]-itv[0] >= _duration_thr])

    _dist_thr = [dist_thr] if isinstance(dist_thr, int) else dist_thr
    assert len(_dist_thr) <= 2

    # post-process
    check = True
    dist_thr_inds = _dist_thr[0] / model_fs
    while check:
        check = False
        rpeaks_diff = np.diff(rpeaks)
        for r in range(len(rpeaks_diff)):
            if rpeaks_diff[r] < dist_thr_inds:  # 200 ms
                prev_r_ind = int(rpeaks[r]/model_granularity)  # ind in _prob
                next_r_ind = int(rpeaks[r+1]/model_granularity)  # ind in _prob
                if _prob[prev_r_ind] > _prob[next_r_ind]:
                    rpeaks = np.delete(rpeaks, r+1)
                    check = True
                    break
                else:
                    rpeaks = np.delete(rpeaks, r)
                    check = True
                    break
    if len(_dist_thr) == 1:
        return rpeaks
    check = True
    # further search should be performed to locate where the
    # distances are greater than 1200 ms between adjacent QRS complexes
    # if there exists at least one point that is great than 0.5, 
    # the threshold of the duration of clustering positive samples is reduced by 16 ms 
    # and this process will continue until a new QRS candidate is found 
    # or the threshold decreases to zero
    check = True
    dist_thr_inds = _dist_thr[1] / model_fs
    while check:
        check = False
        rpeaks_diff = np.diff(rpeaks)
        for r in range(len(rpeaks_diff)):
            if rpeaks_diff[r] >= dist_thr_inds:  # 1200 ms
                prev_r_ind = int(rpeaks[r]/model_granularity)  # ind in _prob
                next_r_ind = int(rpeaks[r+1]/model_granularity)  # ind in _prob
                prev_qrs = [itv for itv in qrs_intervals if itv[0]<=prev_r_ind<=itv[1]][0]
                next_qrs = [itv for itv in qrs_intervals if itv[0]<=next_r_ind<=itv[1]][0]
                check_itv = [prev_qrs[1], next_qrs[0]]
                l_new_itv = mask_to_intervals(mask[check_itv[0]: check_itv[1]], 1)
                if len(l_new_itv) == 0:
                    continue
                l_new_itv = [[itv[0]+check_itv[0], itv[1]+check_itv[0]] for itv in l_new_itv]
                new_itv = max(l_new_itv, key=lambda itv: itv[1]-itv[0])
                new_max_prob = (_prob[new_itv[0]:new_itv[1]]).max()
                for itv in l_new_itv:
                    itv_prob = (_prob[itv[0]:itv[1]]).max()
                    if itv[1] - itv[0] == new_itv[1] - new_itv[0] and itv_prob > new_max_prob:
                        new_itv = itv
                        new_max_prob = itv_prob
                rpeaks = np.insert(rpeaks, r+1, 4*(new_itv[0]+new_itv[1]))
                check = True
                break
    return rpeaks


def _remove_spikes_naive(sig:np.ndarray) -> np.ndarray:
    """ finished, checked,

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
