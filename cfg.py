"""
"""
import os
from easydict import EasyDict as ED


__all__ = [
    "PreprocessCfg",
    "TrainCfg",
]


PreprocessCfg = ED()
PreprocessCfg.rsmp_fs = 400  # Hz, CPSC data fs
PreprocessCfg.remove_baseline = True
# 200 ms and 600 ms ref. (TODO)
PreprocessCfg.baseline_window1 = 80  # corr. to 200 ms
PreprocessCfg.baseline_window2 = 240  # corr. to 600 ms
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TrainCfg.model_path = os.path.join(BASE_DIR, "models", "ecg_xgboost.pkl")
TrainCfg.test_rec_num = 2
TrainCfg.max_iter = 10
TrainCfg.training_data = os.path.join(BASE_DIR, "training_data")
TrainCfg.training_workdir = os.path.join(BASE_DIR, "training_workdir")
