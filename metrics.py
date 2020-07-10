"""
"""
from typing import Union, Optional, Any, List, Tuple

import numpy as np

import utils
from cfg import TrainCfg


__all__ = [
    "CPSC2020_loss",
    "CPSC2020_score",
]


def CPSC2020_loss(y_true:np.ndarray, y_pred:np.ndarray, y_indices:np.ndarray, dtype:type=str, class_weight:Union[str,List[float],np.ndarray,dict]='balanced', verbose:int=0) -> int:
    """ NOT finished, need more consideration!

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

    true_positive = {c: 0 for c in classes}

    for c in classes:
        for tc in truth_arr[c]:
            pc = np.where(abs(pred_arr[c]-tc) <= TrainCfg.bias_thr)[0]
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


def CPSC2020_score(sbp_true:List[np.ndarray], pvc_true:List[np.ndarray], sbp_pred:List[np.ndarray], pvc_pred:List[np.ndarray], verbose:int=0) -> Tuple[int]:
    """
    Score Function for all (test) records

    Parameters:
    -----------
    sbp_true, pvc_true, sbp_pred, pvc_pred: list of ndarray,

    Returns:
    --------
    Score1: int, score for S
    Score2: int, score for V
    """
    s_score = np.zeros([len(sbp_true), ])
    v_score = np.zeros([len(sbp_true), ])
    ## Scoring ##
    for i, s_ref in enumerate(sbp_true):
        v_ref = pvc_true[i]
        s_pos = sbp_pred[i]
        v_pos = pvc_pred[i]
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos-ans) <= TrainCfg.bias_thr)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
                    s_fp += len(s_pos_cand) - 1
        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos-ans) <= TrainCfg.bias_thr)[0]
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

    return Score1, Score2
