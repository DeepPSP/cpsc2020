"""
data generator for feeding data into pytorch models
"""
import os, sys
import json
from random import shuffle, randint
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from easydict import EasyDict as ED
from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

from .cfg import TrainCfg, ModelCfg
from .data_reader import CPSC2020Reader as CR
from .utils import dict_to_str

if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2020",
]


class CPSC2020(Dataset):
    """
    """
    def __init__(self, config:ED, training:bool=True) -> NoReturn:
        """
        """
        raise NotImplementedError

    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        """
        raise NotImplementedError
