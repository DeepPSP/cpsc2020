"""
"""
import os

from sklearn.utils.class_weight import compute_class_weight
from easydict import EasyDict as ED

from utils import CPSC_STATS


__all__ = [
    "PreprocessCfg",
    "FeatureCfg",
    "TrainCfg",
]


#--------------------------------------------------------------
PreprocessCfg = ED()
PreprocessCfg.fs = 400  # Hz, CPSC2020 data fs
PreprocessCfg.preproc = ['baseline', 'bandpass',]  # sequential, keep correct ordering
# for 200 ms and 600 ms, ref. (`ecg_classification` in `reference`)
PreprocessCfg.baseline_window1 = int(0.2*PreprocessCfg.fs)  # 200 ms window
PreprocessCfg.baseline_window2 = int(0.6*PreprocessCfg.fs)  # 600 ms window
PreprocessCfg.filter_band = [0.5, 45]
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
FeatureCfg.morph_intervals = [[0,45], [85,95], [110,120], [170,200]]
FeatureCfg.label_map = dict(N=0,S=1,V=2)
FeatureCfg.beat_ann_bias_thr = 0.1*FeatureCfg.fs  # half width of broad qrs complex


#--------------------------------------------------------------
TrainCfg = ED()
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TrainCfg.fs = PreprocessCfg.fs
TrainCfg.model_path = ED({
    "ml": os.path.join(_BASE_DIR, "models", "ecg_ml.pkl"),
    "dl": os.path.join(_BASE_DIR, "models", "ecg_dl.pkl"),
})
TrainCfg.SEED = 42
TrainCfg.verbose = 1
TrainCfg.label_map = FeatureCfg.label_map
TrainCfg.test_rec_num = 1
TrainCfg.augment_rpeaks = True
TrainCfg.preproc = PreprocessCfg.preproc
TrainCfg.features = FeatureCfg.features
TrainCfg.bias_thr = 0.15*TrainCfg.fs  # keep the same with `THR` in `CPSC202_score.py`
TrainCfg.class_weight = dict(N=0.018,S=1,V=0.42)  # via sklearn.utils.class_weight.compute_class_weight
# TrainCfg.class_weight = 'balanced'
TrainCfg.training_data = os.path.join(_BASE_DIR, "training_data")
TrainCfg.training_workdir = os.path.join(_BASE_DIR, "training_workdir")
TrainCfg.cv = 3
TrainCfg.xgb_native_cv_params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective': 'multi:softmax',
    'num_class': 3,
}
TrainCfg.xgb_native_cv_kw = {
    'num_boost_round': 999,
    'early_stopping_rounds': 20,
    'seed': TrainCfg.SEED,
    'nfold': TrainCfg.cv,
    'metrics': ['merror',],  # Exact matching error, used to evaluate multi-class classification
    'verbose_eval': TrainCfg.verbose,
}
TrainCfg.ml_init_params = {
    'XGBClassifier': 'objective="multi:softmax", num_class=3, verbosity=TrainCfg.verbose',
    'RandomForestClassifier': 'class_weight="balanced", verbosity=TrainCfg.verbose',
    'GradientBoostingClassifier': 'verbosity=TrainCfg.verbose',
    'KNeighborsClassifier': 'verbosity=TrainCfg.verbose',
    'MLPClassifier': 'verbosity=TrainCfg.verbose',
}
TrainCfg.ml_param_grid = {
    'XGBClassifier': {
        # https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
        # 'objective': ['multi:softmax'],
        # 'num_classes': [3],
        "learning_rate": [0.05, 0.10, 0.20, 0.30],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
    },
    # 'SVC': {
    #     'C': [0.0005, 0.001, 0.002, 0.01, 0.1, 1, 10],
    #     'gamma' : [0.001, 0.01, 0.1, 1],
    #     'kernel': ['rbf', 'poly', 'sigmoid']
    # },  # might be too slow
    'RandomForestClassifier': {
        'n_estimators': [10, 40, 70, 100],
        'max_depth': [3, 5, 7],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1, 2],
        'max_features': [0.2, 0.5, 1, 2],
    },
    'GradientBoostingClassifier': {
        'learning_rate': [0.05, 0.1, 0.3],
        'n_estimators': [40, 70, 100],
        'subsample': [0.3, 0.5, 0.7, 1],
        'min_samples_split': [0.2, 0.5, 0.7, 2],
        'min_samples_leaf': [0.2, 0.5, 1],
        'max_depth': [3, 7],
        # 'max_features': [1, 2],
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3,5,7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2, 3, 4, 5],
    },
    'MLPClassifier': {
        'hidden_layer_sizes': [(200,), (300,), (400,)],
        'alpha': [0.001, 0.005, 0.01],
        'batch_size': [128, 256, 512, 1024],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [200, 300, 400, 500],
    },
    # 'BaggingClassifier': {
    #     'n_estimators': [10, 30, 50, 60],
    #     'max_samples': [0.1, 0.3, 0.5, 0.8, 1.],
    #     'max_features': [0.2, 0.5, 1, 2],
    # },
}
