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
from functools import partial
from copy import deepcopy
from typing import Union, Optional, Any, List

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
from .training_data_generator import CPSC2020
import misc


__all__ = ["train"]


class ECGPrematureDetector(object):
    """
    """
    def __init__(self, model:Any, db_dir:str, working_dir:Optional[str]=None, config:Optional[ED]=None, verbose:int=2, **kwargs):
        """

        NOTE: the official scoring functions is
            -1 * 

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
        if isinstance(model, str):
            if model == "XGBClassifier":
                self.model = XGBClassifier(objective="multi:softmax", num_classes=3)
            else:
                self.model = eval(f"{model}()")
            self.model_name = model
        else:
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
            best_model=grid_result.best_estimator_
            best_params=grid_result.best_params_,
            best_score=grid_result.best_score_
        )
        return retval


def CPSC2020_loss(y_true:np.ndarray, y_pred:np.ndarray, dtype:type=str, class_weight:Union[str,List[float],np.ndarray,dict]='balanced') -> int:
    """ NOT finished, need more consideration!

    """
    # valid_intervals = misc.intervals_union([[s-TrainCfg.bias_thr, s+TrainCfg.bias_thr] for s in y_true])
    # temporarily use the official scoring function
    if dtype == str:
        sbp_true = np.where(y_true=='S')[0]
        pvc_true = np.where(y_true=='V')[0]
        sbp_pred = np.where(y_pred=='S')[0]
        pvc_pred = np.where(y_pred=='V')[0]
        sbp_wt = class_weight['S']
        pvc_wt = class_weight['V']
    elif dtype == int:
        sbp_true = np.where(y_true==TrainCfg.label_map['S'])[0]
        pvc_true = np.where(y_true==TrainCfg.label_map['V'])[0]
        sbp_pred = np.where(y_pred==TrainCfg.label_map['S'])[0]
        pvc_pred = np.where(y_pred==TrainCfg.label_map['V'])[0]
        sbp_wt = class_weight[label_map['S']]
        pvc_wt = class_weight[label_map['V']]
    
    sbp_score, pvc_score = CPSC2020_score(
        [sbp_true], [pvc_true], [sbp_pred], [pvc_pred],
    )

    return sbp_score * sbp_wt + pvc_score * pvc_wt



def CPSC2020_score(sbp_true:List[np.ndarray], pvc_true:List[np.ndarray], sbp_pred:List[np.ndarray], pvc_pred:List[np.ndarray]) -> Tuple[int]:
    """
    Score Function for all (test) records

    Parameters:
    -----------
    sbp_true, pvc_true, sbp_pred, pvc_pred: list of ndarray,

    Returns:
    --------
    Score1: int, score for S
    Score2: int, score for V
    """
    s_score = np.zeros([len(sbp_true), ])
    v_score = np.zeros([len(sbp_true), ])
    ## Scoring ##
    for i, s_ref in enumerate(sbp_true):
        v_ref = pvc_true[i]
        s_pos = sbp_pred[i]
        v_pos = pvc_pred[i]
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos-ans) <= THR*FS)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
                    s_fp += len(s_pos_cand) - 1
        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos-ans) <= THR*FS)[0]
                if v_pos_cand.size == 0:
                    v_fn += 1
                else:
                    v_tp += 1
                    v_fp += len(v_pos_cand) - 1
        # calculate the score
        s_score[i] = s_fp * (-1) + s_fn * (-5)
        v_score[i] = v_fp * (-1) + v_fn * (-5)
    Score1 = np.sum(s_score)
    Score2 = np.sum(v_score)

    return Score1, Score2


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
