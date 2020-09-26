"""
"""
import os
import pickle, joblib
from copy import deepcopy
from typing import Union, Optional, Any

import xgboost as xgb

from cfg import TrainCfg


__all__ = ["load_model"]


def load_model(field:str='ml', model_path:Optional[str]=None) -> Union[xgb.Booster,dict]:
    """ finished, checked,

    Parameters:
    -----------
    field: str,
        model type, machine learning ('ml') or deep learning ('dl')
    model_path: str, optional,
        custom model path,
        if not given, default model will be loaded
    
    Returns:
    --------
    model: Booster, dict, etc.
        for machine learning, a Booster,
        or a dict containing 'model' and 'feature_scaler' and perhaps more metadata;
        for deep learning, to implement later...
    """
    if field.lower() in ['ml', 'machine_learning']:
        if model_path:
            print("loading custom machine learning model...")
            mp = deepcopy(model_path)
        else:
            print("loading default machine learning model...")
            mp = TrainCfg.model_in_use['ml']
        model_file_ext = os.path.splitext(mp)[1]
        if model_file_ext == '.bst':
            model = xgb.Booster()
            model.load_model(mp)
        elif model_file_ext == '.pkl':
            with open(mp, "rb") as model_file:
                model = pickle.load(model_file)
        else:
            raise NotImplementedError
    elif field.lower() in ['dl', 'deep_learning']:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model
