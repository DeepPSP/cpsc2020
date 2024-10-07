"""
"""

import os
import sys

# for DAS training ModuleNotFoundError:
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT_DIR)
import argparse
from copy import deepcopy

from easydict import EasyDict as ED
from keras import Input, layers
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.initializers import Orthogonal, he_normal, he_uniform
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    AveragePooling1D,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    CuDNNGRU,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LeakyReLU,
    MaxPooling1D,
    ReLU,
    Reshape,
    TimeDistributed,
    add,
    concatenate,
)
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD, Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cfg import TrainCfg
from dataset import CPSC2020Reader
from signal_processing.ecg_preproc import parallel_preprocess_signal

__all__ = [
    "train",
    "TI_CNN",
    "ATI_CNN",
]


class TI_CNN(Sequential):
    """ """

    def __init__(self, classes: list, input_len: int, cnn: str, bidirectional: bool = True):
        """ """
        super(Sequential, self).__init__(name="TI_CNN")
        self.classes = classes
        self.nb_classes = len(classes)
        self.input_len = input_len
        self.bidirectional = bidirectional
        self.cnn = cnn.lower()

        if self.cnn == "vgg":
            self._build_vgg_model()
        elif self.cnn == "resnet":
            self._build_resnet_model()
        elif self.cnn == "xception":
            self._build_xception_model()

    def _build_vgg_model(self):
        """ """
        raise NotImplementedError

    def _build_resnet_model(self):
        """ """
        raise NotImplementedError


class ATI_CNN(TI_CNN):
    """ """

    def __init__(self, classes: list, input_len: int, cnn: str, bidirectional: bool = True):
        """ """
        super(Sequential, self).__init__(name="TI_CNN")
        self.classes = classes
        self.nb_classes = len(classes)
        self.input_len = input_len
        self.bidirectional = bidirectional
        self.cnn = cnn.lower()


def train(**config):
    """
    NOT finished
    """
    cfg = deepcopy(TrainCfg)
    cfg.update(config or {})
    verbose = cfg.get("verbose", 0)

    data_gen = CPSC2020Reader(db_dir=cfg.db_dir, working_dir=cfg.training_workdir)
