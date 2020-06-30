"""
preprocess of (single lead) ecg signal:
    band pass () --> remove baseline --> 
"""
import multiprocessing as mp
from numbers import Real
from typing import Optional, List

import numpy as np
from wfdb.processing.qrs import XQRS, GQRS, xqrs_detect
from scipy.ndimage.filters import median_filter
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


def preprocess_signal(raw_ecg:np.ndarray, fs:Real) -> np.ndarray:
    """
    """
    filtered_ecg = raw_ecg.copy()

    # remove baseline
    if PreprocessCfg.remove_baseline:
        window1 = 2 * (PreprocessCfg.baseline_window1 // 2) + 1  # window size must be odd
        window2 = 2 * (PreprocessCfg.baseline_window2 // 2) + 1
        baseline = median_filter(self.ecg_curve, size=window1, mode='nearest')
        baseline = median_filter(baseline, size=window2, mode='nearest')
        filtered_ecg = filtered_ecg - baseline
    
    # filter signal
    if PreprocessCfg.filter_signal:
        filtered_ecg = filter_signal(
            signal=filtered_ecg,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * fs),
            sampling_rate=fs,
            frequency=PreprocessCfg.filter_band,
        )['signal']
    
    return filtered_ecg
    

def parallel_preprocess_signal(raw_ecg:np.ndarray, fs:Real, save_path:Optional[str]=None) -> np.ndarray:
    """
    """
    epoch_len = int(PreprocessCfg.parallel_len * fs)
    epoch_overlap = 2 * (int(PreprocessCfg.parallel_overpal * fs) // 2)
    epoch_forward = epoch_len - epoch_overlap

    if len(raw_ecg) <= 5 * epoch_len:
        return preprocess_signal(raw_ecg, fs)
    
    l_epoch = [
        raw_ecg[idx*epoch_forward: (idx+1)*epoch_forward] \
            for idx in range((len(raw_ecg)-epoch_overlap)//epoch_forward - 1)
    ]

    cpu_num = max(1, mp.cpu_count()-6)
    with mp.Pool(processes=cpu_num) as pool:
        result = pool.starmap(
            preprocess_signal, [(e, fs) for e in l_epoch]
        )
    
    filtered_ecg = result[0][:epoch_len-epoch_overlap//2].tolist()
    for e in result[1:]:
        filtered_ecg += e[epoch_overlap//2: -epoch_overlap//2].tolist()
    filtered_ecg = np.array(filtered_ecg)

    if save_path:
        np.save(save_path, filtered_ecg)

    return filtered_ecg
