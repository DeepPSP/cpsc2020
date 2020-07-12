"""
denoise, mainly concerning the motion artefacts

some of the CPSC2020 records have segments of severe motion artefacts,
such segments should be eliminated from feature computation

References:
-----------
to add
"""
from typing import Union, Optional

import numpy as np


__all__ = [
    "ecg_denoise",
]


def ecg_denoise(filtered_sig:np.ndarray):
    """
    """
    raise NotImplementedError
