"""
Possible methods:
-----------------
1. beat-wise detection using handcrafted features and machine learning methods
2. 10s segments 3-classes classification using C(R)NN models, plus post-processing to locate the premature beats
3. premature beats detection using LSTM with RR interval sequence as input, plus post-processing to distinguish ventricular ones from supraventricular ones
4. sequence labeling (or even "segmentation" using UNet) model
5. more?
"""

import os
import sys
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)


# reference submodules

_BIOSPPY_BASE_DIR = os.path.join(_BASE_DIR, "references", "biosppy")
try:
    import biosppy
except ModuleNotFoundError:
    sys.path.insert(0, _BIOSPPY_BASE_DIR)

_ECG_CLASSIFICATION_BASE_DIR = os.path.join(_BASE_DIR, "references", "ecg_classification")
sys.path.insert(0, _ECG_CLASSIFICATION_BASE_DIR)
