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

from tqdm import tqdm
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    make_scorer,
    accuracy_score, fbeta_score, jaccard_score,
    plot_confusion_matrix,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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

        Parameters:
        -----------
        model: classifiers from sklearn or xgboost,
            e.g. `xgb.XGBClassifier()`
        db_dir: str,
            directory where the database is stored
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        self.model = model
        self.model_name = type(self.model).__name__
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
            self._train_xgb_clf(**cfg)
        else:
            self._train_sklearn_clf(**cfg)

    def _train_xgb_clf(self, **config):
        """
        NOT finished
        """
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test)

        cv_results = xgb.cv(
            config.ml_params_grid,
            dtrain,
            num_boost_round=num_boost_round,
            seed=config.SEED,
            nfold=config.cv,
            metrics={'mae'},
            early_stopping_rounds=10
        )

        raise NotImplementedError

    def _train_sklearn_clf(self, **config):
        """
        NOT finished
        """
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=config.ml_params_grid[self.modle],
            scoring=make_scorer(acc),
            n_jobs=n_jobs,
            verbose=self.verbose,
            cv=config.cv,
        )
        grid_result = grid.fit(self.X_train, self.y_train)


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
