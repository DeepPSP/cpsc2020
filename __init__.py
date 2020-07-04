"""
"""
import os, sys


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
