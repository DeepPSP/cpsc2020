"""
NOTE:
    the CPSC data is highly unbalanced,
    with most beats normal beats

References:
-----------
[1] https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
[2] https://github.com/x4nth055/emotion-recognition-using-speech
[3] https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html

TODO:
    1. feature selection,
    2. consider whether features should be normalized using sklearn.preprocessing.StandardScaler
    3. adjust metric function,
       with reference to the official scoring function,
       which lay more punishment on false negatives (5 times)
    4. (?necessary) add AF detection models which depends only on RR intervals,
       or mainly on RR intervals, with auxiliary detector based on wave (f-wave) delineation,
       so that SPB is not confused with AF.
"""
import os, sys
# for DAS training ModuleNotFoundError:
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT_DIR)
import argparse
import joblib, pickle
import time, datetime
import multiprocessing as mp
from functools import partial
from copy import deepcopy
from typing import Union, Optional, Any, List, Tuple, NoReturn

import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from metrics import CPSC2020_loss, CPSC2020_score
import utils


__all__ = [
    "ECGPrematureDetector",
]


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

        self.gpu = kwargs.get("gpu", False)

        if isinstance(model, str):
            assert model.lower() in _CLF_FULL_NAME.keys(), f"model {model} not supported!"
            self.model_name = _CLF_FULL_NAME[model.lower()]
            if self.model_name == "XGBClassifier" and self.gpu:
                self.model = eval(f"{self.model_name}({TrainCfg.xgbc_gpu_init_params})")
            else:
                self.model = eval(f"{self.model_name}({TrainCfg.ml_init_params[self.model_name]})")
        else:
            self.model = model
            self.model_name = type(self.model).__name__

        self.config = deepcopy(TrainCfg)
        self.config.update(config or {})

        self.x_train, self.y_train, self.y_indices_train = None, None, None
        self.x_test, self.y_test, self.y_indices_test = None, None, None
        self.sample_weight = None
        if self.config.feature_scaler:
            self.feature_scaler = eval(f"{self.config.feature_scaler}()")
        else:
            self.feature_scaler = None
        self.fit_params = ED()


    def train_test_split(self, test_rec_num:Optional[int]=None, int_labels:bool=True) -> NoReturn:
        """
        """
        self.x_train, self.y_train, self.y_indices_train, \
        self.x_test, self.y_test, self.y_indices_test = \
            self.data_gen.train_test_split_data(
                test_rec_num=(test_rec_num or self.config.test_rec_num),
                features=self.config.features,
                preproc=self.config.preproc,
                augment=self.config.augment_rpeaks,
                int_labels=int_labels,
            )
        if self.feature_scaler:
            self.feature_scaler.fit(self.x_train)
            self.x_train = self.feature_scaler.transform(self.x_train)
            self.x_test = self.feature_scaler.transform(self.x_test)

        if self.verbose >= 1:
            print(f"self.x_train.shape = {self.x_train.shape}")
            print(f"self.y_train.shape = {self.y_train.shape}")
            print(f"self.y_indices_train.shape = {self.y_indices_train.shape}")
            print(f"self.x_test.shape = {self.x_test.shape}")
            print(f"self.y_test.shape = {self.y_test.shape}")
            print(f"self.y_indices_test.shape = {self.y_indices_test.shape}")
            print(f"feature_scaler.mean = {self.feature_scaler.mean_}")

        if int_labels:
            class_weight = {self.config.label_map[k]: v for k,v in self.config.class_weight.items()}
        else:
            class_weight = self.config.class_weight

        self.sample_weight = ED(
            train=utils.class_weight_to_sample_weight(self.y_train, class_weight),
            test=utils.class_weight_to_sample_weight(self.y_test, class_weight),
        )


    def train(self, config:Optional[ED]=None):
        """ NOT finished

        Parameters:
        -----------
        config: dict, optional,
            extra configurations for training,
            for flexibility of different experiments,
            if set, `self.config` will be updated by this `config`
        """
        if not all([len(self.x_train), len(self.y_train), len(self.y_indices_train), len(self.x_test), len(self.y_test), len(self.y_indices_test)]):
            raise ValueError("do train test split first!")

        self.fit_params = ED({
            "XGBClassifier": {
                "sample_weight": self.sample_weight.train,
                "eval_set": [(self.x_test, self.y_test)],
                "sample_weight_eval_set": [self.sample_weight.test],
            },
            "SVC": {},
            "RandomForestClassifier": {},
            "GradientBoostingClassifier": {},
            "KNeighborsClassifier": {},
            "MLPClassifier": {},
        })

        cfg = deepcopy(self.config)
        cfg.update(config)

        for k,v in cfg.ml_fit_params.items():
            v.update(self.fit_params[k])

        # if type(self.model).__name__ == "XGBClassifier":
        #     self._train_xgb_clf(cfg)
        # else:
        #     self._train_sklearn_clf(cfg)

        grid = GridSearchCV(
            estimator=self.model,
            param_grid=cfg.ml_param_grid[self.model_name],
            # TODO: better scoring function
            # scoring=make_scorer(partial(accuracy_score, sample_weight=self.sample_weight)),
            # n_jobs=max(1, mp.cpu_count()-3),
            n_jobs=-1,
            verbose=self.verbose,
            cv=cfg.cv,
        )

        grid_result = grid.fit(
            self.x_train, self.y_train,
            **cfg.ml_fit_params[self.model_name]
        )

        retval = ED(
            best_model=grid_result.best_estimator_,
            best_params=grid_result.best_params_,
            best_score=grid_result.best_score_,
        )

        return retval


    # def _train_sklearn_clf(self, config:dict):
    #     """ NOT finished,

    #     Parameters:
    #     -----------
    #     config: dict,
    #         configurations for training xgboost classifier,
    #     """
    #     raise NotImplementedError


    def _cv_xgb(self, params:dict):
        """
        """
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train, weight=self.sample_weight.train)
        cv_results = xgb.cv(
            params,
            dtrain,
            **TrainCfg.xgb_native_cv_kw,
        )
        return cv_results


    def train_das_gpu_xgb(self, config:Optional[ED]=None, **kwargs):
        """ NOT finished,

        Parameters:
        -----------
        config: dict,
            configurations for training xgboost classifier,
        """
        cfg = deepcopy(TrainCfg)
        cfg.update(config or {})
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train, weight=self.sample_weight.train)
        dtest = xgb.DMatrix(self.x_test, label=self.y_test, weight=self.sample_weight.test)

        params = {
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'tree_method': 'gpu_hist',
        }
        params.update(cfg.xgb_native_train_params)

        watchlist = [
            (dtest, "Test"),
            (dtrain, "Train"),
        ]
        evals_result = dict()

        start = time.time()
        booster = xgb.train(
            params, dtrain,
            evals=[(dtest, "Test")],
            evals_result=evals_result,
            **config.xgb_native_train_kw,
        )
        print(f"XGB training on DAS GPU costs {(time.time()-start)/60:.2f} minutes")
        print(f"evals_result = {utils.dict_to_str(evals_result)}")

        save_path_params = '_'.join([str(k)+'-'+str(v) for k,v in params.items()])
        scaler_name = type(self.feature_scaler).__name__ if self.feature_scaler else 'no-scaler'
        save_path = cfg.model_path['ml'].format(
            model_name=self.model_name,
            time=utils.get_date_str(),
            params=save_path_params,
            scaler=scaler_name,
            ext='pkl',
        )
        # booster.save_model(save_path)
        save_dict = {
            'feature_scaler': self.feature_scaler,
            'model': booster,
        }

        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)


DAS = True

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="extra training setting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        '-m', '--model',
        type=str, default='xgbc',
        help='name of the model, separated by ","',
        dest='models',
    )
    ap.add_argument(
        "-d", "--db-dir",
        # type=str, required=True,
        type=str, default="/mnt/wenhao71/data/CPSC2020/TrainingSet/",
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
        '-g', '--gpu',
        type=utils.str2bool,
        default=True,
        help='use gpu (only for xgboost) or not',
        dest='gpu',
    )
    ap.add_argument(
        '-l', '--lr',
        type=float, default=0.1,
        help='learning rate of xgb booster',
        dest='lr',
    )
    ap.add_argument(
        '-v', '--verbose',
        type=int, default=0,
        help='set verbosity',
        dest='verbose',
    )
    kw = vars(ap.parse_args())

    nl = "\n"
    print(f"passed keyword arguments:{nl}{utils.dict_to_str(kw)}")

    models = kw.pop("models")
    models = list(map(lambda m: _CLF_FULL_NAME[m], models.split(",")))
    lr = kw.pop("lr")
    # verbose = kw.pop("verbose")
    # db_dir = kw.pop("db_dir")
    # working_dir = kw.pop("working_dir")
    # gpu = kw.pop("gpu")

    config = deepcopy(TrainCfg)
    # config.update(kw)

    for m in models:
        trainer = ECGPrematureDetector(
            model=m, **kw
            # db_dir=db_dir,
            # working_dir=working_dir,
            # config=config,
            # verbose=verbose,
            # gpu=gpu,
        )  # NOT finished
        trainer.train_test_split(test_rec_num=config.test_rec_num,int_labels=True)
        if DAS:
            trainer.train_das_gpu_xgb(config, learning_rate=lr)
        else:
            trainer.train(config)
