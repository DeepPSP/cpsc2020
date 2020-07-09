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
import multiprocessing as mp
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
    def __init__(self, model:Any, db_dir:str, working_dir:Optional[str]=None, config:Optional[ED]=None, verbose:int=2, **kwargs):
        """

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
        self.model = model
        self.model_name = type(self.model).__name__
        self.db_dir = db_dir
        self.working_dir = working_dir or os.getcwd()
        self.verbose = verbose
        self.data_gen = CPSC2020(
            db_dir=db_dir, working_dir=working_dir, verbose=verbose
        )
        self.config = deepcopy(TrainCfg)
        self.config.update(config or {})
        self.x_train, self.y_train, self.x_test, self.y_test = \
            self.data_gen.train_test_split_data(
                test_rec_num=self.config.test_rec_num,
                features=self.config.features,
                preprocesses=self.config.preprocesses,
                augment=self.config.augment_rpeaks,
                int_labels=True,
            )
        self.sample_weight = class_weight_to_sample_weight(self.y_train, self.config.class_weight)

    def train(self, **config):
        """
        NOT finished

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
        """
        NOT finished
        """
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train, weight=self.sample_weight)
        # dtest = xgb.DMatrix(self.x_test, label=self.y_test)

        cv_results = xgb.cv(
            config.ml_param_grid[self.modle_name],
            dtrain,
            num_boost_round=num_boost_round,
            seed=config.SEED,
            nfold=config.cv,
            metrics={''},
            # early_stopping_rounds=10,
        )

        raise NotImplementedError

    def _train_sklearn_clf(self, config:dict):
        """
        NOT finished
        """
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=config.ml_param_grid[self.modle_name],
            scoring=make_scorer(accuracy_score),
            n_jobs=max(1, mp.cpu_count()-3),
            verbose=self.verbose,
            cv=config.cv,
        )
        grid_result = grid.fit(self.X_train, self.y_train)
        retval = ED(
            best_model=grid_result.best_estimator_
            best_params=grid_result.best_params_,
            best_score=grid_result.best_score_
        )
        return retval


def class_weight_to_sample_weight(y:np.ndarray, class_weight:Union[str,List[float],np.ndarray,dict]='balanced') -> np.ndarray:
    """ finished, checked,

    transform class weight to sample weight

    Parameters:
    -----------
    y: ndarray,
        the label (class) of each sample
    class_weight: str, or list, or ndarray, or dict, default 'balanced',
        the weight for each sample class,
        if is 'balanced', the class weight will automatically be given by 
        if `y` is of string type, then `class_weight` should be a dict,
        if `y` is of numeric type, and `class_weight` is array_like,
        then the labels (`y`) should be continuous and start from 0
    """
    if not class_weight:
        sample_weight = np.ones_like(y, dtype=float)
        return sample_weight
    
    try:
        sample_weight = y.copy().astype(int)
    except:
        sample_weight = y.copy()
        assert isinstance(class_weight, dict) or class_weight.lower()=='balanced', \
            "if `y` are of type str, then class_weight should be 'balanced' or a dict"
    
    if class_weight.lower() == 'balanced':
        classes = np.unique(y).tolist()
        cw = compute_class_weight('balanced', classes=classes, y=y)
        trans_func = lambda s: cw[classes.index(s)]
    else:
        trans_func = lambda s: class_weight[s]
    sample_weight = np.vectorize(trans_func)(sample_weight)
    sample_weight = sample_weight / np.max(sample_weight)
    return sample_weight





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
