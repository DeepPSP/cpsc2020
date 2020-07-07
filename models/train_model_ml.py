"""
References:
-----------
[1] https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
[2] https://github.com/x4nth055/emotion-recognition-using-speech
"""
import os
import argparse
import joblib, pickle
from copy import deepcopy
from typing import Union, Optional, Any

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from easydict import EasyDict as ED

from cfg import TrainCfg
from signal_processing.ecg_preprocess import parallel_preprocess_signal
from signal_processing.ecg_features import compute_ecg_features
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


def train(**config):
    """
    NOT finished
    """
    cfg = deepcopy(TrainCfg)
    cfg.update(config or {})
    verbose = cfg.get("verbose", 0)
    
    data_gen = CPSC2020(db_dir=cfg.training_data, working_dir=cfg.training_workdir)
    # x_train, y_train, x_test, y_test = data_gen.train_test_split_data(cfg.test_rec_num, config)


    # joblib.dump(model, config.model_path.ml)


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
