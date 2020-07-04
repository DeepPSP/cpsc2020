"""
preprocess of (single lead) ecg signal:
    band pass () --> remove baseline --> find rpeaks

References:
-----------
[1] https://github.com/PIA-Group/BioSPPy
[2] to add
"""
import os
import multiprocessing as mp
from copy import deepcopy
from numbers import Real
from typing import Optional, List, Dict

import numpy as np
from easydict import EasyDict as ED
from wfdb.processing.qrs import XQRS, GQRS, xqrs_detect, gqrs_detect
from wfdb.processing.pantompkins import pantompkins as _pantompkins
from scipy.ndimage.filters import median_filter
from scipy.signal.signaltools import resample
from scipy.io import savemat
# from scipy.signal import medfilt
# https://github.com/scipy/scipy/issues/9680
try:
    from biosppy.signals.tools import filter_signal
except:
    from references.biosppy.biosppy.signals.tools import filter_signal

from cfg import PreprocessCfg


__all__ = [
    "preprocess_signal",
    "parallel_preprocess_signal",
]


def pantompkins(sig, fs, *args, **kwargs):
    """ to keep in accordance of parameters with `xqrs` and `gqrs` """
    return _pantompkins(sig, fs)


QRS_DETECTORS = {
    "xqrs": xqrs_detect,
    "gqrs": gqrs_detect,
    "pantompkins": pantompkins,
}


def preprocess_signal(raw_ecg:np.ndarray, fs:Real, config:Optional[ED]=None) -> Dict[str, np.ndarray]:
    """

    Parameters:
    -----------
    raw_ecg: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_ecg`
    config: dict, optional,
        extra process configuration,
        `PreprocessCfg` will `update` this `config`

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if 'rpeaks' in `config` is not set
    """
    filtered_ecg = raw_ecg.copy()

    cfg = deepcopy(PreprocessCfg)
    cfg.update(config or {})

    if fs != cfg.rsmp_fs:
        filtered_ecg = resample(filtered_ecg, int(round(len(filtered_ecg)*PreprocessCfg.fs/fs)))

    # remove baseline
    if cfg.remove_baseline:
        window1 = 2 * (cfg.baseline_window1 // 2) + 1  # window size must be odd
        window2 = 2 * (cfg.baseline_window2 // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode='nearest')
        baseline = median_filter(baseline, size=window2, mode='nearest')
        filtered_ecg = filtered_ecg - baseline
    
    # filter signal
    if cfg.filter_signal:
        filtered_ecg = filter_signal(
            signal=filtered_ecg,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * fs),
            sampling_rate=fs,
            frequency=cfg.filter_band,
        )['signal']

    if cfg.rpeaks:
        detector = QRS_DETECTORS[cfg.rpeaks.lower()]
        rpeaks = detector(sig=filtered_ecg, fs=fs)
    else:
        rpeaks = np.array([], dtype=int)

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })
    
    return retval
    

def parallel_preprocess_signal(raw_ecg:np.ndarray, fs:Real, config:Optional[ED]=None, save_dir:Optional[str]=None, save_fmt:str='npy') -> Dict[str, np.ndarray]:
    """

    Parameters:
    -----------
    raw_ecg: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_ecg`
    config: dict, optional,
        extra process configuration,
        `PreprocessCfg` will `update` this `config`
    save_dir: str, optional,
        directory for saving the outcome ('filtered_ecg' and 'rpeaks')
    save_fmt: str, default 'npy',
        format of the save files, 'npy' or 'mat'

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if 'rpeaks' in `config` is not set
    """
    cfg = deepcopy(PreprocessCfg)
    cfg.update(config or {})

    epoch_len = int(cfg.parallel_len * fs)
    epoch_overlap_half = int(cfg.parallel_overlap * fs) // 2
    epoch_overlap = 2 * epoch_overlap_half
    epoch_forward = epoch_len - epoch_overlap

    if len(raw_ecg) <= 3 * epoch_len:  # too short, no need for parallel computing
        return preprocess_signal(raw_ecg, fs, cfg)
    
    l_epoch = [
        raw_ecg[idx*epoch_forward: idx*epoch_forward + epoch_len] \
            for idx in range((len(raw_ecg)-epoch_overlap)//epoch_forward)
    ]

    if cfg.parallel_keep_tail:
        tail_start_idx = epoch_forward * len(l_epoch) + epoch_overlap
        if len(raw_ecg) - tail_start_idx < 30 * fs:  # less than 30s, make configurable?
            # append to the last epoch
            l_epoch[-1] = np.append(l_epoch[-1], raw_ecg[tail_start_idx:])
        else:  # long enough
            tail_epoch = raw_ecg[tail_start_idx-epoch_overlap:]
            l_epoch.append(tail_epoch)

    cpu_num = max(1, mp.cpu_count()-3)
    with mp.Pool(processes=cpu_num) as pool:
        result = pool.starmap(
            preprocess_signal, [(e, fs, cfg) for e in l_epoch]
        )

    if cfg.parallel_keep_tail:
        tail_result = result[-1]
        result = result[:-1]
    
    filtered_ecg = result[0]['filtered_ecg'][:epoch_len-epoch_overlap_half]
    rpeaks = result[0]['rpeaks'][np.where(result[0]['rpeaks']<epoch_len-epoch_overlap_half)[0]]
    for idx, e in enumerate(result[1:]):
        filtered_ecg = np.append(filtered_ecg, e['filtered_ecg'][epoch_overlap_half: -epoch_overlap_half])
        epoch_rpeaks = e['rpeaks'][np.where( (e['rpeaks'] >= epoch_overlap_half) & (e['rpeaks'] < epoch_len-epoch_overlap_half) )[0]]
        rpeaks = np.append(rpeaks, (idx+1)*epoch_forward + epoch_rpeaks)

    if cfg.parallel_keep_tail:
        filtered_ecg = np.append(filtered_ecg, tail_result['filtered_ecg'][epoch_overlap_half:])
        tail_rpeaks = tail_result['rpeaks'][np.where(tail_result['rpeaks'] >= epoch_overlap_half)[0]]
        rpeaks = np.append(rpeaks, len(result)*epoch_forward + epoch_rpeaks)

    if save_dir:
        # NOTE: this part is not tested
        os.makedirs(save_dir, exist_ok=True)
        if save_fmt.lower() == 'npy':
            np.save(os.path.join(save_dir, "filtered_ecg.npy"), filtered_ecg)
            np.save(os.path.join(save_dir, "rpeaks.npy"), rpeaks)
        elif save_fmt.lower() == 'mat':
            # save into 2 files, keep in accordance
            savemat(os.path.join(save_dir, "filtered_ecg.mat"), {"filtered_ecg": filtered_ecg}, format='5')
            savemat(os.path.join(save_dir, "rpeaks.mat"), {"rpeaks": rpeaks}, format='5')

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })

    return retval
