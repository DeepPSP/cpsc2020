"""
"""
import argparse
from copy import deepcopy

from keras import layers
from keras import Input
from keras.models import Sequential, Model, load_model
from keras.layers import (
    Layer,
    LSTM, GRU,
    TimeDistributed, Bidirectional,
    ReLU, LeakyReLU,
    BatchNormalization,
    Dense, Dropout, Activation, Flatten, 
    Input, Reshape, GRU, CuDNNGRU,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D, AveragePooling1D,
    concatenate, add,
)
from keras.initializers import he_normal, he_uniform, Orthogonal
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error
from easydict import EasyDict as ED

from cfg import TrainCfg
from signal_processing.ecg_preprocess import parallel_preprocess_signal
from .data_generator import CPSC2020


__all__ = [
    "train",
    "TI_CNN", "ATI_CNN",
]


class TI_CNN(Sequential):
    """
    """
    def __init__(self, classes:list, input_len:int, cnn:str, bidirectional:bool=True):
        """
        """
        super(Sequential, self).__init__(name='TI_CNN')
        self.classes = classes
        self.nb_classes = len(classes)
        self.input_len = input_len
        self.bidirectional = bidirectional
        self.cnn = cnn.lower()

        if self.cnn == 'vgg':
            self._build_vgg_model()
        elif self.cnn == 'resnet':
            self._build_resnet_model()
        elif self.cnn == 'xception':
            self._build_xception_model()

    def _build_vgg_model(self):
        """
        """
        raise NotImplementedError

    def _build_resnet_model(self):
        """
        """
        raise NotImplementedError


class ATI_CNN(TI_CNN):
    """
    """
    def __init__(self, classes:list, input_len:int, cnn:str, bidirectional:bool=True):
        """
        """
        super(Sequential, self).__init__(name='TI_CNN')
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

    data_gen = CPSC2020(db_dir=cfg.training_data, working_dir=cfg.training_workdir)
    
