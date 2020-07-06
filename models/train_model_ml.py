"""
References:
-----------
[1] https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f
"""
import argparse
import joblib, pickle
from copy import deepcopy
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from easydict import EasyDict as ED

from cfg import TrainCfg
from signal_processing.ecg_preprocess import parallel_preprocess_signal
from signal_processing.ecg_features import compute_ecg_features
from .training_data_generator import CPSC2020


__all__ = ["train"]


def train(**config):
    """
    NOT finished
    """
    cfg = deepcopy(TrainCfg)
    cfg.update(config or {})
    verbose = cfg.get("verbose", 0)
    
    data_gen = CPSC2020(db_dir=TrainCfg.training_data, working_dir=TrainCfg.training_workdir)
    train_records, test_records = data_gen.train_test_split(TrainCfg.test_rec_num)
    
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    # params_grid = {
    #     'max_depth':6,
    #     'min_child_weight': 1,
    #     'learning_rate':.3,
    #     'subsample': 1,
    #     'colsample_bytree': 1,
    # }

    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        seed=42,
        nfold=5,
        metrics={'mae'},
        early_stopping_rounds=10
    )

    # n_iter = 0
    # best_rmse = 1.0
    # best_ratio = 1.0
    # model = None
    # min_n_estimator = 7
    # n_estimators = [min_n_estimator]  # TODO

    # while n_iter < cfg.max_iter:
    #     if verbose >= 1:
    #         print('while loop entered')
    #         print(f'best_rmse = {best_rmse}, best_ratio = {best_ratio}, n_iter = {n_iter}')
    #     param_grid = {'max_depth': [4,6,8,10], 'n_estimators': n_estimators}
    #     base_model = xgb.XGBClassifier()
    #     grid_search = GridSearchCV(base_model, param_grid, verbose=verbose)
    #     grid_search.fit(X, y)
    #     best_model = grid_search.best_estimator_
    #     y_pred = best_model.predict(X)
    #     rmse = np.sqrt(mean_squared_error(y_pred, y))

    #     if verbose >= 2:
    #         if 'plt' not in dir():
    #             import matplotlib.pyplot as plt
    #         fig, ax = plt.subplots()
    #         ax.plot(np.arange(1,10,1), np.arange(1,10,1), '.--')
    #         ax.plot(y, y_pred, '.')
    #         ax.axhline(y=3.9, linestyle='dashed', color='r')
    #         ax.axvline(x=3.9, linestyle='dashed', color='r')
    #         ax.set_xlabel('Truth')
    #         ax.set_ylabel('Predicted')
    #         ax.text(6, 1.6, f'RMSE: {int(round(rmse,5))}', color='red', fontsize=12, style='italic')
    #         plt.show()

    #     cm = confusion_matrix(y, y_pred,labels=[True,False])
    #     false_negative = cm[0][1]
    #     true_positive = cm[0][0]
    #     ratio = false_negative / true_positive

    #     if verbose >= 1:
    #         print(f'RMSE = {rmse}')
    #         print(f'true_positive: {true_positive}')
    #         print(f'false_negative: {false_negative}')
    #         print(*best_model.get_params().items(), sep='\n')

    #     if rmse >= 0.1:
    #         if rmse < best_rmse:
    #             best_rmse = rmse
    #             best_ratio = ratio
    #             model = best_model
    #         if rmse < 0.5 and ratio <= 0.1:
    #             break
    #         else: # underfit, enlarge n_estimators
    #             n_estimators = [n+5 for n in n_estimators]  # TODO
    #     else: # overfit, shrink n_estimators
    #         n_estimators = [max(min_n_estimator, n-2) for n in n_estimators]  # TODO
    #     n_iter += 1

    if model is None:
        model = best_model

    joblib.dump(model, config.model_path)


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
