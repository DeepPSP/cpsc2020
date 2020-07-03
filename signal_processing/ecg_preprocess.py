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
from scipy.ndimage.filters import median_filter
from scipy.signal.signaltools import resample
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


QRS_DETECTORS = {
    "xqrs": xqrs_detect,
    "gqrs", gqrs_detect,
}


def preprocess_signal(raw_ecg:np.ndarray, fs:Real, config:Optional[ED]=None) -> Dict[str, np.ndarray]:
    """

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    filtered_ecg = raw_ecg.copy()

    config = deepcopy(PreprocessCfg).update(config or {})

    if fs != config.fs:
        filtered_ecg = resample(filtered_ecg, int(round(len(filtered_ecg)*PreprocessCfg.fs/fs)))

    # remove baseline
    if config.remove_baseline:
        window1 = 2 * (config.baseline_window1 // 2) + 1  # window size must be odd
        window2 = 2 * (config.baseline_window2 // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode='nearest')
        baseline = median_filter(baseline, size=window2, mode='nearest')
        filtered_ecg = filtered_ecg - baseline
    
    # filter signal
    if config.filter_signal:
        filtered_ecg = filter_signal(
            signal=filtered_ecg,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * fs),
            sampling_rate=fs,
            frequency=config.filter_band,
        )['signal']

    if config.rpeaks:
        detector = QRS_DETECTORS[config.rpeaks.lower()]
        rpeaks = detector(sig=filtered_ecg, fs=fs)
    else:
        rpeaks = np.array([], dtype=int)

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })
    
    return retval
    

def parallel_preprocess_signal(raw_ecg:np.ndarray, fs:Real, config:Optional[ED]=None, save_dir:Optional[str]=None) -> Dict[str, np.ndarray]:
    """

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    config = deepcopy(PreprocessCfg).update(config or {})

    epoch_len = int(config.parallel_len * fs)
    epoch_overlap = 2 * (int(config.parallel_overpal * fs) // 2)
    epoch_forward = epoch_len - epoch_overlap

    if len(raw_ecg) <= 5 * epoch_len:
        return preprocess_signal(raw_ecg, fs, config)
    
    l_epoch = [
        raw_ecg[idx*epoch_forward: (idx+1)*epoch_forward] \
            for idx in range((len(raw_ecg)-epoch_overlap)//epoch_forward - 1)
    ]

    cpu_num = max(1, mp.cpu_count()-6)
    with mp.Pool(processes=cpu_num) as pool:
        result = pool.starmap(
            preprocess_signal, [(e, fs, config) for e in l_epoch]
        )
    
    filtered_ecg = result[0]['filtered_ecg'][:epoch_len-epoch_overlap//2]
    rpeaks = result[0]['rpeaks'][np.where(result[0]['rpeaks']<epoch_len-epoch_overlap//2)[0]]
    for idx, e in enumerate(result[1:]):
        filtered_ecg = np.append(filtered_ecg, e['filtered_ecg'][epoch_overlap//2: -epoch_overlap//2])
        epoch_rpeaks = e['rpeaks'][np.where( (e['rpeaks']>=epoch_overlap//2) & (e['rpeaks']<epoch_len-epoch_overlap//2) )[0]]
        rpeaks = np.append(rpeaks, (idx+1)*epoch_forward + epoch_rpeaks)

    if save_dir:
        np.save(os.path.join(save_dir, "filtered_ecg.npy"), filtered_ecg)
        np.save(os.path.join(save_dir, "rpeak.npy"), rpeaks)

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })

    return retval
