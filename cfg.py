"""
"""
import os
from copy import deepcopy

import pywt
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
# BaseCfg.training_data = os.path.join(_BASE_DIR, "training_data")
BaseCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2020/TrainingSet/"

BaseCfg.bias_thr = 0.15 * BaseCfg.fs  # keep the same with `THR` in `CPSC202_score.py`
BaseCfg.beat_ann_bias_thr = 0.1 * BaseCfg.fs  # half width of broad qrs complex
BaseCfg.beat_winL = 250 * BaseCfg.fs // 1000  # corr. to 250 ms
BaseCfg.beat_winR = 250 * BaseCfg.fs // 1000  # corr. to 250 ms

BaseCfg.torch_dtype = "float"  # "double"


PreprocCfg = ED()
PreprocCfg.fs = BaseCfg.fs
# sequential, keep correct ordering, to add 'motion_artefact'
PreprocCfg.preproc = ['bandpass',]  # 'baseline',
# for 200 ms and 600 ms, ref. (`ecg_classification` in `reference`)
PreprocCfg.baseline_window1 = int(0.2*PreprocCfg.fs)  # 200 ms window
PreprocCfg.baseline_window2 = int(0.6*PreprocCfg.fs)  # 600 ms window
PreprocCfg.filter_band = [0.5, 45]
PreprocCfg.parallel_epoch_len = 600  # second
PreprocCfg.parallel_epoch_overlap = 10  # second
PreprocCfg.parallel_keep_tail = True
PreprocCfg.rpeaks = 'xqrs'  # TODO: use deep learning models ?
# or 'gqrs', or 'pantompkins', 'hamilton', 'ssf', 'christov', 'engzee', 'gamboa'
# or empty string '' if not detecting rpeaks
"""
for qrs detectors:
    `xqrs` sometimes detects s peak (valley) as r peak,
    but according to Jeethan, `xqrs` has the best performance
"""


# FeatureCfg only for ML models, deprecated
FeatureCfg = ED()
FeatureCfg.fs = BaseCfg.fs
FeatureCfg.features = ['wavelet', 'rr', 'morph',]

FeatureCfg.wt_family = 'db1'
FeatureCfg.wt_level = 3
FeatureCfg.beat_winL = BaseCfg.beat_winL
FeatureCfg.beat_winR = BaseCfg.beat_winR
FeatureCfg.wt_feature_len = pywt.wavedecn_shapes(
    shape=(1+FeatureCfg.beat_winL+FeatureCfg.beat_winR,), 
    wavelet=FeatureCfg.wt_family,
    level=FeatureCfg.wt_level
)[0][0]

FeatureCfg.rr_local_range = 10  # 10 r peaks
FeatureCfg.rr_global_range = 5 * 60 * FeatureCfg.fs  # 5min, units in number of points
FeatureCfg.rr_normalize_radius = 30  # number of beats (rpeaks)

FeatureCfg.morph_intervals = [[0,45], [85,95], [110,120], [170,200]]



ModelCfg = ED()
ModelCfg.fs = BaseCfg.fs
ModelCfg.torch_dtype = BaseCfg.torch_dtype


TrainCfg = ED()
TrainCfg.fs = ModelCfg.fs
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.input_len = int(10 * TrainCfg.fs)  # 10 s
TrainCfg.overlap_len = int(8 * TrainCfg.fs)  # 8 s
TrainCfg.normalize_data = True

# data augmentation
TrainCfg.flip = True  # signal upside down
TrainCfg.label_smoothing = 0.1
TrainCfg.random_mask = int(TrainCfg.fs * 0.0)  # 1.0s, 0 for no masking
TrainCfg.stretch_compress = 1.0  # stretch or compress in time axis
# TODO: add more data augmentation
TrainCfg.gaussian_std = 0.05  # gaussian noise, with mean 0; if std = 0, gaussian noise not added
# sinusoidal signal with random initial phase and amplitude
# randomly shifting the baseline
TrainCfg.flip = True  # making the signal upside down

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 128
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd"

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

TrainCfg.lr_scheduler = None  # 'plateau', 'burn_in', 'step', None
