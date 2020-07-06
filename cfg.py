"""
"""
import os
from easydict import EasyDict as ED


__all__ = [
    "PreprocessCfg",
    "FeatureCfg",
    "TrainCfg",
]


#--------------------------------------------------------------
PreprocessCfg = ED()
PreprocessCfg.fs = 400  # Hz, CPSC2020 data fs
PreprocessCfg.preprocesses = ['baseline', 'bandpass',]
# PreprocessCfg.remove_baseline = True
# for 200 ms and 600 ms, ref. (`ecg_classification` in `reference`)
PreprocessCfg.baseline_window1 = 80  # corr. to 200 ms
PreprocessCfg.baseline_window2 = 240  # corr. to 600 ms
# PreprocessCfg.filter_signal = True
PreprocessCfg.filter_band = [0.5,45]
PreprocessCfg.parallel_epoch_len = 600  # second
PreprocessCfg.parallel_epoch_overlap = 10  # second
PreprocessCfg.parallel_keep_tail = True
PreprocessCfg.rpeaks = 'xqrs'  # or 'gqrs', or 'pantompkins' or empty string ''
"""
for qrs detectors:
    `xqrs` sometimes detects s peak (valley) as r peak,
    but according to Jeethan, `xqrs` has the best performance
"""


#--------------------------------------------------------------
FeatureCfg = ED()
FeatureCfg.fs = PreprocessCfg.fs  # Hz, CPSC2020 data fs
FeatureCfg.beat_winL = 100  # corr. to 250 ms
FeatureCfg.beat_winR = 100  # corr. to 250 ms
FeatureCfg.features = ['wavelet', 'rr', 'morph',]
FeatureCfg.wt_family = 'db1'
FeatureCfg.wt_level = 3
FeatureCfg.rr_local_range = 10  # 10 r peaks
FeatureCfg.rr_global_range = 5*60*FeatureCfg.fs  # 5min, units in number of points
FeatureCfg.label_map = dict(N=0,S=1,V=2)
FeatureCfg.beat_ann_bias_thr = int(0.15*PreprocessCfg.fs)


#--------------------------------------------------------------
TrainCfg = ED()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TrainCfg.fs = PreprocessCfg.fs
TrainCfg.model_path = ED({
    "ml": os.path.join(BASE_DIR, "models", "ecg_ml.pkl"),
    "dl": os.path.join(BASE_DIR, "models", "ecg_dl.pkl"),
})
TrainCfg.SEED = 42
TrainCfg.label_map = FeatureCfg.label_map
TrainCfg.test_rec_num = 2
TrainCfg.bias_thr = int(0.15*TrainCfg.fs)  # keep the same with `THR` in `CPSC202_score.py`
TrainCfg.max_iter = 10
TrainCfg.training_data = os.path.join(BASE_DIR, "training_data")
TrainCfg.training_workdir = os.path.join(BASE_DIR, "training_workdir")
