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
        model_path = TrainCfg.model_in_use['ml']
        model_file_ext = os.path.splitext(model_path)[1]
        if model_file_ext == '.bst':
            model = xgb.Booster()
            model.load_model(model_path)
        elif model_file_ext == '.pkl':
            with open(model_path, "rb") as model_file:
                model = pickle.load(model_file)
        else:
            raise NotImplementedError
    elif field.lower() in ['dl', 'deep_learning']:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
