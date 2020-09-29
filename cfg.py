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
BaseCfg.label_map = dict(N=0, S=1, V=2)
BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")

BaseCfg.bias_thr = 0.15 * BaseCfg.fs  # keep the same with `THR` in `CPSC202_score.py`
BaseCfg.beat_ann_bias_thr = 0.1 * BaseCfg.fs  # half width of broad qrs complex
BaseCfg.beat_winL = 100  # corr. to 250 ms
BaseCfg.beat_winR = 100  # corr. to 250 ms


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
FeatureCfg.features = ['wavelet', 'rr', 'morph',]

FeatureCfg.wt_family = 'db1'
FeatureCfg.wt_level = 3
FeatureCfg.wt_feature_len = pywt.wavedecn_shapes(
    shape=(1+FeatureCfg.beat_winL+FeatureCfg.beat_winR,), 
    wavelet=FeatureCfg.wt_family,
    level=FeatureCfg.wt_level
)[0][0]

FeatureCfg.rr_local_range = 10  # 10 r peaks
FeatureCfg.rr_global_range = 5*60*FeatureCfg.fs  # 5min, units in number of points

FeatureCfg.morph_intervals = [[0,45], [85,95], [110,120], [170,200]]



ModelCfg = ED()



TrainCfg = ED()
