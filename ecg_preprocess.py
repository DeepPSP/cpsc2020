"""
preprocess of (single lead) ecg signal:
    band pass () --> remove baseline --> 
"""
from numbers import Real
from typing import List

import numpy as np
from wfdb.processing.qrs import XQRS, GQRS, xqrs_detect
try:
    from biosppy.signals.tools import filter_signal
except:
    from references.biosppy.biosppy.signals.tools import filter_signal


__all__ = [
    "preprocess_signal",
]


def preprocess_signal(raw_ecg:np.ndarray, fs:Real, band:List[Real, Real]=[0.5,45]) -> np.ndarray:
    """
    """
    # band pass
    filtered_ecg = filter_signal(
        signal=raw_ecg,
        ftype='FIR',
        band='bandpass',
        order=int(0.3 * fs),
        sampling_rate=fs,
        frequency=band,
    )
