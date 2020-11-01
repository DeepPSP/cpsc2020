"""
"""
from numbers import Real
from typing import Tuple

import numpy as np
from scipy.signal import resample_poly

from signal_processing.ecg_preproc import parallel_preprocess_signal
from signal_processing.ecg_denoise import ecg_denoise
from saved_models import load_model
from cfg import ModelCfg


CRNN_MODEL, SEQ_LAB_MODEL = load_model(which="both")
CRNN_CFG, SEQ_LAB_CFG = ModelCfg.crnn, ModelCfg.seq_lab


def CPSC2020_challenge(ECG, fs):
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
    # finished, NOT checked,
    
    S_pos_rsmp, V_pos_rsmp = np.array([], dtype=int), np.array([], dtype=int)

    FS = 400

    if int(fs) != FS:
        sig = resample_poly(np.array(ECG).flatten(), up=FS, down=int(fs))
    else:
        sig = np.array(ECG).flatten()
    pps = parallel_preprocess_signal(sig, fs)  # use default config in `cfg`
    filtered_ecg = pps['filtered_ecg']
    rpeaks = pps['rpeaks']
    valid_intervals = ecg_denoise(filtered_ecg, fs=FS, config={"ampl_min":0.15})
    rpeaks = [r for r in rpeaks if any([itv[0]<=r<=itv[1] for itv in valid_intervals])]

    # classify and sequence labeling models

    seq_lab_granularity = 8
    model_input_len = 10 * FS  # 10s
    half_overlap_len = 512  # should be divisible by `model_granularity`
    overlap_len = 2 * half_overlap_len
    forward_len = model_input_len - overlap_len

    n_segs, residue = divmod(len(filtered_ecg)-overlap_len, forward_len)
    if residue != 0:
        filtered_ecg = np.append(filtered_ecg, np.zeros((forward_len-residue,)))
        n_segs += 1
    batch_size = 64
    n_batches = math.ceil(n_segs / batch_size)

    MEAN, STD = 0.01, 0.25  # rescale to this mean and standard deviation
    segs = list(range(n_segs))
    for b_idx in range(n_batches):
        b_start = b_idx * batch_size
        b_segs = segs[b_start: b_start + batch_size]
        b_input = []
        for idx in b_segs:
            seg = filtered_ecg[idx*forward_len: idx*forward_len+model_input_len]
            if np.std(seg) > 0:
                seg = (seg - np.mean(seg) + MEAN) / np.std(seg) * STD
            b_input.append(seg)
        b_input = np.vstack(b_input).reshape((-1, 1, model_input_len))
        
        _, crnn_out = \
            CRNN_MODEL.inference(b_input, bin_pred_thr=0.5)  # (batch_size, 3)
        _, SPB_indices, PVC_indices = \
            SEQ_LAB_MODEL.inference(b_input, bin_pred_thr=0.5)

        for i, idx in enumerate(b_segs):
            if crnn_out[i, CRNN_CFG.index("N")] == 1:
                # the classifier predicts non-premature segment
                continue
            if crnn_out[i, CRNN_CFG.index("S")] == 1:
                seg_spb = np.array(SPB_indices[i])
                seg_spb = seg_spb[np.where((seg_spb>=half_overlap_len) & (seg_spb<model_input_len-half_overlap_len)[0]] + idx * forward_len
                S_pos_rsmp = np.append(S_pos_rsmp, seg_spb)
            if crnn_out[i, CRNN_CFG.index("V")] == 1:
                seg_pvc = np.array(PVC_indices[i])
                seg_pvc = seg_pvc[np.where((seg_pvc>=half_overlap_len) & (seg_pvc<model_input_len-half_overlap_len)[0]] + idx * forward_len
                V_pos_rsmp = np.append(V_pos_rsmp, seg_pvc)

    S_pos_rsmp = S_pos_rsmp[np.where(S_pos_rsmp<len(filtered_ecg))[0]]
    V_pos_rsmp = V_pos_rsmp[np.where(V_pos_rsmp<len(filtered_ecg))[0]]

    if int(fs) != FS:
        S_pos = np.round(S_pos_rsmp * fs / FS).astype(int)
        V_pos = np.round(V_pos_rsmp * fs / FS).astype(int)
    else:
        S_pos, V_pos = S_pos_rsmp, V_pos_rsmp    

    return S_pos, V_pos
