"""
"""
from numbers import Real
from typing import Union, Optional, Any, List, Tuple

import numpy as np
from easydict import EasyDict as ED

from . import utils
from .cfg import BaseCfg


__all__ = [
    "CPSC2020_loss",
    "CPSC2020_score",
]


def CPSC2020_loss(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str, verbose:int=0) -> int:
    """ finished, updated with the latest (updated on 2020.8.31) official function

    Parameters:
    -----------
    y_true: ndarray,
        array of ground truth of beat types
    y_true: ndarray,
        array of predictions of beat types
    y_indices: ndarray,
        indices of beat (rpeak) in the original ecg signal
    dtype: type, default str,
        dtype of `y_true` and `y_pred`

    Returns:
    --------
    total_loss: int,
        the total loss of all ectopic beat types (SBP, PVC)
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
            truth_arr[c] = y_indices[np.where(y_true==BaseCfg.label_map[c])[0]]
            pred_arr[c] = y_indices[np.where(y_pred==BaseCfg.label_map[c])[0]]

    true_positive = {c: 0 for c in classes}

    for c in classes:
        for tc in truth_arr[c]:
            pc = np.where(abs(pred_arr[c]-tc) <= BaseCfg.bias_thr)[0]
            if pc.size > 0:
                true_positive[c] += 1

    false_positive = {
        c: len(pred_arr[c]) - true_positive[c] for c in classes
    }
    false_negative = {
        c: len(truth_arr[c]) - true_positive[c] for c in classes
    }

    false_positive_loss = {c: 1 for c in classes}
    false_negative_loss = {c: 5 for c in classes}

    if verbose >= 1:
        print(f"true_positive = {utils.dict_to_str(true_positive)}")
        print(f"false_positive = {utils.dict_to_str(false_positive)}")
        print(f"false_negative = {utils.dict_to_str(false_negative)}")

    total_loss = sum([
        false_positive[c] * false_positive_loss[c] + false_negative[c] * false_negative_loss[c] \
            for c in classes
    ])
    
    return total_loss


def CPSC2020_score(sbp_true:List[np.ndarray], pvc_true:List[np.ndarray], sbp_pred:List[np.ndarray], pvc_pred:List[np.ndarray], verbose:int=0) -> Union[Tuple[int],dict]:
    """
    Score Function for all (test) records

    Parameters:
    -----------
    sbp_true, pvc_true, sbp_pred, pvc_pred: list of ndarray,
    verbose: int

    Returns:
    --------
    retval: tuple or dict,
        tuple of (negative) scores for each ectopic beat type (SBP, PVC), or
        dict of more scoring details, including
        - total_loss: sum of loss of each ectopic beat type (PVC and SPB)
        - true_positive: number of true positives of each ectopic beat type
        - false_positive: number of false positives of each ectopic beat type
        - false_negative: number of false negatives of each ectopic beat type
    """
    s_score = np.zeros([len(sbp_true), ], dtype=int)
    v_score = np.zeros([len(sbp_true), ], dtype=int)
    ## Scoring ##
    for i, (s_ref, v_ref, s_pos, v_pos) in enumerate(zip(sbp_true, pvc_true, sbp_pred, pvc_pred)):
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        # SBP
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos-ans) <= BaseCfg.bias_thr)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
                    s_fp += len(s_pos_cand) - 1
        # PVC
        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos-ans) <= BaseCfg.bias_thr)[0]
                if v_pos_cand.size == 0:
                    v_fn += 1
                else:
                    v_tp += 1
                    v_fp += len(v_pos_cand) - 1
        # calculate the score
        s_score[i] = s_fp * (-1) + s_fn * (-5)
        v_score[i] = v_fp * (-1) + v_fn * (-5)

        if verbose >= 1:
            print(f"for the {i}-th record")
            print(f"s_tp = {s_tp}, s_fp = {s_fp}, s_fn = {s_fn}")
            print(f"v_tp = {v_tp}, v_fp = {v_fp}, s_fn = {v_fn}")
            print(f"s_score[{i}] = {s_score[i]}, v_score[{i}] = {v_score[i]}")

    Score1 = np.sum(s_score)
    Score2 = np.sum(v_score)

    if verbose >= 1:
        retval = ED(
            total_loss=-(Score1+Score2),
            class_loss={'S':-Score1, 'V':-Score2},
            true_positive={'S':s_tp, 'V':v_tp},
            false_positive={'S':s_fp, 'V':v_fp},
            false_negative={'S':s_fn, 'V':v_fn},
        )
    else:
        retval = Score1, Score2

    return retval
