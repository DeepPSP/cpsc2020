"""
References:
-----------
[1] https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
[2] https://github.com/x4nth055/emotion-recognition-using-speech
[3] https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

TODO:
    adjust metric function,
    with reference to the official scoring function,
    which lay more punishment on false negatives (5 times)
"""
import os
import argparse
import joblib, pickle
import multiprocessing as mp
from functools import partial
from copy import deepcopy
from typing import Union, Optional, Any, List, Tuple

import numpy as np
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
from xgboost import XGBClassifier
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
from dataset import CPSC2020
import misc


__all__ = ["train"]


_CLF_FULL_NAME = {
    "xgbc": "XGBClassifier",
    "xgbclassifier": "XGBClassifier",
    # "svc": "SVC",
    "rfc": "RandomForestClassifier",
    "randomforestclassifier": "RandomForestClassifier",
    "gbc": "GradientBoostingClassifier",
    "gradientboostingclassifier": "GradientBoostingClassifier",
    "knn": "KNeighborsClassifier",
    "kneighborsclassifier": "KNeighborsClassifier",
    "mpl": "MLPClassifier",
    "mplclassifier": "MLPClassifier",
}

_ALL_CLF = list(TrainCfg.ml_param_grid.keys())


class ECGPrematureDetector(object):
    """
    """
    def __init__(self, model:Any, db_dir:str, working_dir:Optional[str]=None, config:Optional[ED]=None, verbose:int=1, **kwargs):
        """

        NOTE: the official scoring (-loss) functions is
            (-1) * false_positives + (-5) * false_negatives

        Parameters:
        -----------
        model: classifiers from sklearn or xgboost,
            e.g. `xgb.XGBClassifier()`
        db_dir: str,
            directory where the database is stored
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        config: dict, optional,
            extra configurations for training,
            if set, `TrainCfg` will be updated by this `config`
        verbose: int, default 2,
        """
        self.db_dir = db_dir
        self.working_dir = working_dir or os.getcwd()
        self.verbose = max(TrainCfg.verbose, verbose)
        self.data_gen = CPSC2020(
            db_dir=db_dir, working_dir=working_dir, verbose=verbose
        )

        if isinstance(model, str):
            self.model_name = _CLF_FULL_NAME[model.lower()]
            self.model = eval(f"{self.model_name}({TrainCfg.ml_init_params[self.model_name]})")
        else:
            self.model = model
            self.model_name = type(self.model).__name__
        self.config = deepcopy(TrainCfg)
        self.config.update(config or {})

        self.x_train, self.y_train, self.y_indices_train, self.x_test, self.y_test, self.y_indices_test = \
            self.data_gen.train_test_split_data(
                test_rec_num=self.config.test_rec_num,
                features=self.config.features,
                preprocesses=self.config.preprocesses,
                augment=self.config.augment_rpeaks,
                int_labels=False,
            )
        self.sample_weight = misc.class_weight_to_sample_weight(self.y_train, self.config.class_weight)


    def train(self, **config):
        """ NOT finished

        Parameters:
        -----------
        config: dict, optional,
            extra configurations for training,
            for flexibility of different experiments,
            if set, `self.config` will be updated by this `config`
        """
        cfg = deepcopy(self.config)
        cfg.update(config)

        if type(self.model).__name__ == "XGBClassifier":
            self._train_xgb_clf(cfg)
        else:
            self._train_sklearn_clf(cfg)

    def _train_xgb_clf(self, config:dict):
        """ NOT finished,

        Parameters:
        -----------
        config: dict,
            configurations for training xgboost classifier,
        """
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train, weight=self.sample_weight)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test, weight=self.samplt_weight)
        

        # cpvc_pred = xgb.cv(
        #     config.ml_param_grid[self.modle_name],
        #     dtrain,
        #     num_boost_round=num_boost_round,
        #     seed=config.SEED,
        #     nfold=config.cv,
        #     metrics='merror',  # Exact matching error, used to evaluate multi-class classification
        #     # early_stopping_rounds=10,
        # )

        raise NotImplementedError

    def _train_sklearn_clf(self, config:dict):
        """ NOT finished,

        Parameters:
        -----------
        config: dict,
            configurations for training xgboost classifier,
        """
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=config.ml_param_grid[self.modle_name],
            scoring=make_scorer(partial(accuracy_score, sample_weight=self.sample_weight)),
            n_jobs=max(1, mp.cpu_count()-3),
            verbose=self.verbose,
            cv=config.cv,
        )
        grid_result = grid.fit(self.X_train, self.y_train)
        retval = ED(
            best_model=grid_result.best_estimator_,
            best_params=grid_result.best_params_,
            best_score=grid_result.best_score_,
        )
        return retval


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="extra training setting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '-m', '--model',
        type=str, default='',
        help='name of the model, separated by ","',
        dest='models',
    )
    ap.add_argument(
        "-d", "--db-dir",
        type=str, required=True,
        help="directory where the database is stored",
        dest="db_dir",
    )
    ap.add_argument(
        "-w", "--working-dir",
        type=str, default=None,
        help="working directory",
        dest="working_dir",
    )
    ap.add_argument(
        '-v', '--verbose',
        type=int, default=0,
        help='set verbosity',
        dest='verbose',
    )
    kw = vars(ap.parse_args())
    models = kw.pop("models")
    models = list(map(lambda m: _CLF_FULL_NAME[m], models.split(",")))
    verbose = kw.pop("verbose")
    db_dir = kw.pop("db_dir")
    working_dir = kw.pop("working_dir")

    config = deepcopy(TrainCfg)
    config.update(kw)

    for m in models:
        config["model"] = m
        trainer = ECGPrematureDetector(m, db_dir, working_dir, config, verbose)  # NOT finished
        train(**config)
