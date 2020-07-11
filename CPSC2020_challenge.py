"""
"""
import numpy as np

from cfg import FeatureCfg
from signal_processing.ecg_preprocess import preprocess_signal, parallel_preprocess_signal
from signal_processing.ecg_features import compute_ecg_features
from models.load_model import load_model


def CPSC2020_challenge(ECG, fs=400):
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

#   ====== arrhythmias detection =======

#    S_pos = np.zeros([1, ])
#    V_pos = np.zeros([1, ])
    pr = parallel_preprocess_signal(ECG, fs)  # use default config in `cfg`
    filtered_ecg = pr['filtered_ecg']
    rpeaks = pr['rpeaks']

    features = compute_ecg_features(filtered_ecg, rpeaks)

    model = load_model()
    if model is None:
        model = train()

    raise NotImplementedError
    return S_pos, V_pos
