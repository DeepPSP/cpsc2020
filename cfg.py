"""
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "BaseCfg",
    "PreprocCfg",
    "FeatureCfg",
    "ModelCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


BaseCfg = ED()
BaseCfg.fs = 400  # Hz, CPSC2020 data fs
BaseCfg.bias_thr = 0.15 * BaseCfg.fs  # keep the same with `THR` in `CPSC202_score.py`
BaseCfg.label_map = dict(N=0,S=1,V=2)
BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")


PreprocCfg = ED()
PreprocCfg.fs = BaseCfg.fs
# sequential, keep correct ordering, to add 'motion_artefact'
PreprocCfg.preproc = ['baseline', 'bandpass',]
# for 200 ms and 600 ms, ref. (`ecg_classification` in `reference`)
PreprocCfg.baseline_window1 = int(0.2*PreprocCfg.fs)  # 200 ms window
PreprocCfg.baseline_window2 = int(0.6*PreprocCfg.fs)  # 600 ms window
PreprocCfg.filter_band = [0.5, 45]
PreprocCfg.parallel_epoch_len = 600  # second
PreprocCfg.parallel_epoch_overlap = 10  # second
PreprocCfg.parallel_keep_tail = True
PreprocCfg.rpeaks = 'xqrs'
# or 'gqrs', or 'pantompkins', 'hamilton', 'ssf', 'christov', 'engzee', 'gamboa'
# or empty string '' if not detecting rpeaks
"""
for qrs detectors:
    `xqrs` sometimes detects s peak (valley) as r peak,
    but according to Jeethan, `xqrs` has the best performance
"""


FeatureCfg = ED()


ModelCfg = ED()


TrainCfg = ED()
