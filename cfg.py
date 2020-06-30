"""
"""
from easydict import EasyDict as ED


__all__ = [
    "PreprocessCfg",
    "TrainCfg",
]


PreprocessCfg = ED()
PreprocessCfg.rsmp_fs = 400  # Hz, CPSC data fs
PreprocessCfg.remove_baseline = True
PreprocessCfg.baseline_window1 = 200  # ms
PreprocessCfg.baseline_window2 = 600  # ms
PreprocessCfg.filter_signal = True
PreprocessCfg.filter_band = [0.5,45]
PreprocessCfg.parallel_len = 600  # second
PreprocessCfg.parallel_overlap = 10  # second
PreprocessCfg.rpeaks = 'xqrs'  # or 'gqrs'


FeatureCfg = ED()
FeatureCfg.beat_winL = 100  # corr. to 250 ms
FeatureCfg.beat_winR = 100  # corr. to 250 ms
FeatureCfg.features = ['wavelet', 'rr', 'morph',]
FeatureCfg.wt_family = 'db1'
FeatureCfg.wt_level = 3


TrainCfg = ED()
