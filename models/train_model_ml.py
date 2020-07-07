"""
References:
-----------
[1] https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
[2] https://github.com/x4nth055/emotion-recognition-using-speech
[3] https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
"""
import os
import argparse
import joblib, pickle
from copy import deepcopy
from typing import Union, Optional, Any

import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from easydict import EasyDict as ED

from cfg import TrainCfg
# from signal_processing.ecg_preprocess import parallel_preprocess_signal
# from signal_processing.ecg_features import compute_ecg_features
from .training_data_generator import CPSC2020


__all__ = ["train"]


class ECGPrematureDetector(object):
    """
    """
    def __init__(self, model:Any, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """
        """
        self.model = model
        self.db_dir = db_dir
        self.working_dir = working_dir or os.getcwd()
        self.verbose = verbose
        self.data_gen = CPSC2020(
            db_dir=db_dir, working_dir=working_dir, verbose=verbose
        )
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.data_gen.train_test_split_data(test_rec_num=2)

    def train(self, **config):
        """
        NOT finished
        """
        cfg = deepcopy(TrainCfg)
        cfg.update(config)

        if type(self.model).__name__ == "XGBClassifier":
            self._train_xgb_clf(self)
        raise NotImplementedError

    
    def _train_xgb_clf(self):
        """
        """
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test)

        raise NotImplementedError

    def _train_sklearn_clf(self):
        """
        """
        raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="extra training setting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '-v', '--verbose',
        type=int, default=0,
        help='set verbosity',
        dest='verbose',
    )
    config = deepcopy(TrainCfg)
    config.update(vars(ap.parse_args()))

    train(**config)
