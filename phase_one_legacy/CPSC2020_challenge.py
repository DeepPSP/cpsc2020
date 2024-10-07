"""
"""

from numbers import Real
from typing import Tuple

import numpy as np
import xgboost as xgb
from models.load_model import load_model

import utils
from cfg import FeatureCfg
from signal_processing.ecg_features import compute_ecg_features
from signal_processing.ecg_preproc import parallel_preprocess_signal, preprocess_signal


def CPSC2020_challenge(ECG: np.ndarray, fs: Real = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    % This function can be used for events 1 and 2. Participants are free to modify any
    % components of the code. However the function prototype must stay the same
    % [S_pos,V_pos] = CPSC2020_challenge(ECG,fs) where the inputs and outputs are specified
    % below.
    %
    %% Inputs
    %       ECG : raw ecg vector signal 1-D signal
    %       fs  : sampling rate
    %
    %% Outputs
    %       S_pos : the position where SPBs detected
    %       V_pos : the position where PVCs detected
    %
    %
    %
    % Copyright (C) 2020 Dr. Chengyu Liu
    % Southeast university
    % chengyu@seu.edu.cn
    %
    % Last updated : 02-23-2020

    """

    #    ====== arrhythmias detection =======
    sig = np.array(ECG).copy().flatten()
    pps = parallel_preprocess_signal(sig, fs)  # use default config in `cfg`
    filtered_ecg = pps["filtered_ecg"]
    rpeaks = pps["rpeaks"]
    filtered_rpeaks = rpeaks[np.where((rpeaks >= FeatureCfg.beat_winL) & (rpeaks < len(sig) - FeatureCfg.beat_winR))[0]]

    features = compute_ecg_features(filtered_ecg, filtered_rpeaks)

    model = load_model(field="ml")
    # if model is None:
    #     model = train()

    if isinstance(model, dict):
        if model.get("feature_scaler", None):
            features = model["feature_scaler"].transform(features)
        model = model["model"]

    if type(model).__name__ == "Booster":
        # xgboost native Booster
        y_pred = model.predict(xgb.DMatrix(features))
    else:
        y_pred = model.predict(features)

    S_pos, V_pos = utils.pred_to_indices(y_pred, filtered_rpeaks, class_map=FeatureCfg.class_map)

    return S_pos, V_pos
