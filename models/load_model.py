"""
"""
import joblib
from ..cfg import TrainCfg


__all__ = ["load_model"]


def load_model():
    """
    """
    try:
        model = joblib.load(TrainCfg.model_path)
    except:
        model = None
    return model
