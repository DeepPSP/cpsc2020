"""
"""
from easydict import EasyDict as ED


__all__ = [
    "PreprocessCfg",
    "TrainCfg",
]


PreprocessCfg = ED()
PreprocessCfg.remove_baseline = True
PreprocessCfg.baseline_window1 = 200  # ms
PreprocessCfg.baseline_window2 = 600  # ms
PreprocessCfg.filter_signal = True
PreprocessCfg.filter_band = [0.5,45]
PreprocessCfg.parallel_len = 600  # second
PreprocessCfg.parallel_overlap = 10  # second


TrainCfg = ED()
