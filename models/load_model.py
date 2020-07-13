"""
"""
import os
import pickle, joblib

import xgboost as xgb

from cfg import TrainCfg


__all__ = ["load_model"]


def load_model(field:str='ml'):
    """
    """
    if field.lower() in ['ml', 'machine_learning']:
        model_path = TrainCfg.model_in_use[field.lower()]
        if os.path.splitext(model_path)[1] == 'bst':
            model = xgb.Booster()
            model.load_model(model_path)
        else:
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
    else:
        raise NotImplementedError
    return model
