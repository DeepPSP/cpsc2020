"""
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
from .data_generator import CPSC2020


__all__ = ["train"]


def train(**config):
    """
    NOT finished
    """
    pass
