"""
"""
import os
from typing import Union, Optional, Tuple

try:
    from keras.models import model_from_json, Model
except:
    from tensorflow.keras.models import model_from_json, Model

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


__all__ = [
    "load_model",
]


def load_model(name:str) -> Union[Model, Tuple[Model,...]]:
    """ NOT finished, NOT checked,

    Parameters:
    -----------
    name: str,
        name of the model
    """
    if name.lower() == "ecg_seq_lab_net":
        cnn_model, crnn_model = _load_ecg_seq_lab_net()
        return cnn_model, crnn_model
    else:
        raise NotImplementedError


def _load_ecg_seq_lab_net() -> Tuple[Model,Model]:
    """ NOT finished, NOT checked,
    """
    cnn_config_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.json")
    cnn_h5_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CNN.h5")
    cnn_model = model_from_json(open(cnn_config_path).read())
    cnn_model.load_weights(cnn_h5_path)

    crnn_config_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.json")
    crnn_h5_path = os.path.join(_BASE_DIR, "CPSC2019_0416", "CRNN.h5")
    crnn_model = model_from_json(open(crnn_config_path).read())
    crnn_model.load_weights(crnn_h5_path)

    return cnn_model, crnn_model
