"""

References:
-----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
[2] to add more
"""
import os
from typing import Union, Optional, Tuple

try:
    from keras.models import model_from_json, Model
except:
    from tensorflow.keras.models import model_from_json, Model
import torch
from torch import nn


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


__all__ = [
    "load_model",
]


def load_model(name:str) -> Union[Model, Tuple[Model,...], nn.Module, Tuple[nn.Module,...]]:
    """ finished, checked,

    Parameters:
    -----------
    name: str,
        name of the model
    """
    if name.lower() == "keras_ecg_seq_lab_net":
        cnn_model, crnn_model = _load_keras_ecg_seq_lab_net()
        return cnn_model, crnn_model
    elif name.lower() == "pytorch_ecg_seq_lab_net":
        raise NotImplementedError
    else:
        raise NotImplementedError


def _load_keras_ecg_seq_lab_net() -> Tuple[Model,Model]:
    """ finished, checked,

    load the CNN model and CRNN model from the entry 0416 of CPSC2019
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


def _load_pytorch_ecg_seq_lab_net():
    """
    """
    raise NotImplementedError
