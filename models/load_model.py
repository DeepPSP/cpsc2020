"""
"""
import joblib
from cfg import TrainCfg


__all__ = ["load_model"]


def load_model(field:str='ml'):
    """
    """
    try:
        model = joblib.load(TrainCfg.model_path[field.lower()])
    except:
        model = None
    return model
