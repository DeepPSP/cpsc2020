"""
"""
from typing import Union, Optional, Any, List, Tuple

import numpy as np

import utils
from cfg import TrainCfg
from metrics import CPSC2020_loss, CPSC2020_score


def CPSC2020_loss_v0(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str, class_weight:Union[str,List[float],np.ndarray,dict]='balanced') -> int:
    """ NOT finished, too slow!

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    classes = ['S', 'V']

    truth_arr = {}
    pred_arr = {}
    if dtype == str:
        for c in classes:
            truth_arr[c] = y_indices[np.where(y_true==c)[0]]
            pred_arr[c] = y_indices[np.where(y_pred==c)[0]]
    elif dtype == int:
        for c in classes:
            truth_arr[c] = y_indices[np.where(y_true==TrainCfg.label_map[c])[0]]
            pred_arr[c] = y_indices[np.where(y_pred==TrainCfg.label_map[c])[0]]

    pred_intervals = {
        c: [[idx-TrainCfg.bias_thr, idx+TrainCfg.bias_thr] for idx in pred_arr[c]] \
            for c in classes
    }

    true_positive = {
        c: np.array([utils.in_generalized_interval(idx, pred_intervals[c]) for idx in truth_arr[c]]).astype(int).sum() \
            for c in classes
    }
    false_positive = {
        c: len(pred_arr[c]) - true_positive[c] for c in classes
    }
    false_negative = {
        c: len(truth_arr[c]) - true_positive[c] for c in classes
    }

    false_positive_loss = {c: 1 for c in classes}
    false_negative_loss = {c: 5 for c in classes}

    total_loss = sum([
        false_positive[c] * false_positive_loss[c] + false_negative[c] * false_negative_loss[c] \
            for c in classes
    ])

    return total_loss


def CPSC2020_loss_test(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str, class_weight:Union[str,List[float],np.ndarray,dict]='balanced') -> int:
    """

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    classes = ['S', 'V']

    truth_arr = {}
    pred_arr = {}
    if dtype == str:
        for c in classes:
            truth_arr[c] = y_indices[np.where(y_true==c)[0]]
            pred_arr[c] = y_indices[np.where(y_pred==c)[0]]
    elif dtype == int:
        for c in classes:
            truth_arr[c] = y_indices[np.where(y_true==TrainCfg.label_map[c])[0]]
            pred_arr[c] = y_indices[np.where(y_pred==TrainCfg.label_map[c])[0]]

    scores = CPSC2020_score([truth_arr['S']],[truth_arr['V']],[pred_arr['S']],[pred_arr['V']])

    loss = -sum(scores)

    return loss
