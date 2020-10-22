"""
data generator for feeding data into pytorch models

Augmentations:
--------------
label smoothing (label)
(re-)normalize to random mean and std (signal, offline)
baseline wandering (signal, on the fly)
flip (signal, on the fly)
sinusoidal noise (signal, the same as baseline wandering?)
gaussian noise (signal, offline)
stretch and compress (signal, offline)

References:
-----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
[2] Tan, Jen Hong, et al. "Application of stacked convolutional and long short-term memory network for accurate identification of CAD ECG signals." Computers in biology and medicine 94 (2018): 19-26.
[3] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
"""
import os, sys
import json
from random import shuffle, randint
from copy import deepcopy
from functools import reduce
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from scipy import signal as SS
from easydict import EasyDict as ED
try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

from cfg import TrainCfg, ModelCfg, PreprocCfg
from data_reader import CPSC2020Reader as CR
from signal_processing.ecg_preproc import parallel_preprocess_signal
from utils import (
    dict_to_str, mask_to_intervals,
    gen_gaussian_noise, gen_sinusoidal_noise, gen_baseline_wander,
)

if ModelCfg.torch_dtype.lower() == 'double':
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "CPSC2020",
]


class CPSC2020(Dataset):
    """

    data generator for deep learning models,

    strategy:
    ---------
    1. slice each record into short segments of length `TrainCfg.input_len`,
    and of overlap length `TrainCfg.overlap_len` around premature beats
    2. do augmentations for premature segments
    """
    def __init__(self, config:ED, training:bool=True) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        config: dict,
            configurations for the Dataset,
            ref. `cfg.TrainCfg`
        training: bool, default True,
            if True, the training set will be loaded, otherwise the test set
        """
        super().__init__()
        self.config = deepcopy(config)
        self.reader = CR(db_dir=config.db_dir)
        if ModelCfg.torch_dtype.lower() == 'double':
            self.dtype = np.float64
        else:
            self.dtype = np.float32
        self.allowed_preproc = PreprocCfg.preproc
        self.all_classes = self.config.classes
        self.n_classes = len(self.config.classes)

        # preprocess_dir stores pre-processed signals
        self.preprocess_dir = os.path.join(config.db_dir, "preprocessed")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        # segments_dir for sliced segments
        self.segments_dir = os.path.join(config.db_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)
        # rpeaks_dir for detected r peaks, for optional use
        self.rpeaks_dir = os.path.join(config.db_dir, "rpeaks")
        os.makedirs(self.rpeaks_dir, exist_ok=True)

        # self.segments = 


    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        """
        raise NotImplementedError


    def persistence(self, force_recompute:bool=False) -> NoReturn:
        """ NOT finished, NOT checked,

        make the dataset persistent w.r.t. the ratios in `self.config`

        Parameters:
        -----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        """
        self._preprocess_data(self.allowed_preproc)
        self._slice_data()

    def _preprocess_data(self, preproc:List[str], force_recompute:bool=False) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters:
        -----------
        preproc: list of str,
            type of preprocesses to perform,
            should be sublist of `self.allowed_preproc`
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        """
        preproc = self._normalize_preprocess_names(preproc, True)
        suffix = self._get_rec_suffix(preproc)
        config = deepcopy(PreprocCfg)
        config.preproc = preproc
        save_fp = ED()
        for rec in self.reader.all_records:
            # format save path
            rec_name = self.reader._get_rec_name(rec)
            save_fp.data = os.path.join(self.preprocess_dir, f"{rec_name}-{suffix}{self.reader.rec_ext}")
            save_fp.rpeaks = os.path.join(self.rpeaks_dir, f"{rec_name}-{suffix}{self.reader.rec_ext}")
            if (not force_recompute) and os.path.isdir(save_fp.data) and os.path.isdir(save_fp.rpeaks):
                continue
            # perform pre-process
            pps = parallel_preprocess_signal(self.reader.load_data(rec, keep_dim=False), fs=self.fs, config=config)
            pps['rpeaks'] = pps['rpeaks'][np.where( (pps['rpeaks']>=config.beat_winL) & (pps['rpeaks']<len(pps['filtered_ecg'])-config.beat_winR) )[0]]
            # save mat, keep in accordance with original mat files
            savemat(save_fp.data, {'ecg': np.atleast_2d(pps['filtered_ecg']).T}, format='5')
            savemat(save_fp.rpeaks, {'rpeaks': np.atleast_2d(pps['rpeaks']).T}, format='5')

    def _normalize_preprocess_names(self, preproc:List[str], ensure_nonempty:bool) -> List[str]:
        """ finished, checked

        to transform all preproc into lower case,
        and keep them in a specific ordering 
        
        Parameters:
        -----------
        preproc: list of str,
            list of preprocesses types,
            should be sublist of `self.allowd_features`
        ensure_nonempty: bool,
            if True, when the passed `preproc` is empty,
            `self.allowed_preproc` will be returned

        Returns:
        --------
        _p: list of str,
            'normalized' list of preprocess types
        """
        _p = [item.lower() for item in preproc] if preproc else []
        if ensure_nonempty:
            _p = _p or self.allowed_preproc
        # ensure ordering
        _p = [item for item in self.allowed_preproc if item in _p]
        # assert all([item in self.allowed_preproc for item in _p])
        return _p

    def _get_rec_suffix(self, operations:List[str]) -> str:
        """ finished, checked,

        Parameters:
        -----------
        operations: list of str,
            names of operations to perform (or has performed),
            should be sublist of `self.allowed_preproc`

        Returns:
        --------
        suffix: str,
            suffix of the filename of the preprocessed ecg signal
        """
        suffix = '-'.join(sorted([item.lower() for item in operations]))
        return suffix

    def _slice_data(self, force_recompute:bool=False) -> NoReturn:
        """ NOT finished, NOT checked,
        
        Parameters:
        -----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        """
        for rec in self.reader.all_records:
            rec_name = self.reader._get_rec_name(rec)
            save_fp = os.path.join(self.segments_dir, f"{rec_name}{self.rec_ext}")
            data = self.reader.load_data(rec, units="mV", keep_dim=False)
            ann = self.reader.load_ann(rec)
            border_dist = int(2 * self.config.fs)
            forward_len = self.config.input_len - self.config.overlap_len

            spb_mask = np.zeros((len(data),), dtype=int)
            pvc_mask = np.zeros((len(data),), dtype=int)
            spb_mask[ann["SPB_indices"]] = 1
            pvc_mask[ann["PVC_indices"]] = 1
            # generate initial segments with no overlap for non premature beats
            n_init_seg = len(data)//self.config.input_len
            segments = (data[:self.config.input_len*n_init_seg]).reshape((n_init_seg, self.config.input_len))
            labels = np.zeros((n_init_seg, self.n_classes))
            labels[..., self.config.class_map["N"]] = 1
            # for idx in range(n_init_seg):
            #     start_idx = idx * self.config.input_len
            #     end_idx = start_idx + self.config.input_len
            #     if spb_mask[start_idx:end_idx].any():
            #         labels[idx, self.config.class_map["S"]] = 1
            #         labels[idx, self.config.class_map["N"]] = 0
            #     if pvc_mask[start_idx:end_idx].any():
            #         labels[idx, self.config.class_map["V"]] = 1
            #         labels[idx, self.config.class_map["N"]] = 0
            # leave only non premature segments
            non_premature = np.logical_or(spb_mask, pvc_mask)[:self.config.input_len*n_init_seg]
            non_premature = non_premature.reshape((n_init_seg, self.config.input_len)).sum(axis=1)
            segments = segments[non_premature, ...]
            labels = labels[non_premature, ...]

            # do data augmentation for premature beats
            # first locate all possible premature segments
            # mask for segment start indices
            premature_mask = np.zeros((len(data),), dtype=int)
            for idx in np.concatenate((ann["SPB_indices"], ann["PVC_indices"])):
                start_idx = max(0, idx-self.config.input_len+border_dist)
                end_idx = max(start_idx, min(idx-border_dist, len(data)-self.config.input_len))
                premature_mask[start_idx: end_idx] = 1
            # intervals for allowed start of augmented segments
            premature_intervals = mask_to_intervals(premature_mask, 1)
            for itv in premature_intervals:
                start_idx = itv[0]
                while start_idx < itv[1]:
                    end_idx = start_idx + self.config.input_len
                    new_seg = data[start_idx:end_idx]
                    new_label = np.zeros((self.n_classes,))
                    if spb_mask[start_idx:end_idx].any():
                        new_label[self.config.class_map["S"]] = 1
                    if pvc_mask[start_idx:end_idx].any():
                        new_label[self.config.class_map["V"]] = 1
                    new_label = new_label.reshape((1,-1))
                    segments = np.append(segments, new_seg)
                    labels = np.append(labels, new_label, axis=0)
                    
                    # TODO: perform data augmentation on such segments
                    seg_ampl = np.max(new_seg) - np.min(new_seg)
                    if self.config.bw:
                        for ar in self.config.bw_ampl_ratio:
                            bw_ampl = ar * seg_ampl
                            bw = gen_baseline_wander(
                                siglen=self.config.input_len,
                                fs=self.config.fs,
                                bw_fs=self.config.bw_fs,
                                amplitude=bw_ampl,
                            )
                            aug_seg = (new_seg + bw).reshape((1,-1))
                            segments = np.append(segments, aug_seg, axis=0)
                            labels = np.append(labels, new_label, axis=0)
                    if self.config.gaussian_std > 0:
                        gn = gen_gaussian_noise(
                            siglen=self.config.input_len,
                            mean=0,
                            std=self.config.gaussian_std
                        )
                        aug_seg = (new_seg + gn).reshape((1,-1))
                        segments = np.append(segments, aug_seg, axis=0)
                        labels = np.append(labels, new_label, axis=0)
                    if self.config.stretch_compress != 1:
                        aug_seg = data[start_idx: int(round(self.config.stretch_compress*self.config.input_len))]
                        aug_seg = SS.resample(aug_seg, self.config.input_len).reshape((1,-1))
                        segments = np.append(segments, aug_seg, axis=0)
                        labels = np.append(labels, new_label, axis=0)
                    if self.config.flip:
                        pass

                    start_idx += forward_len
        # randomly shuffle the data
        seg_inds = list(range(segments.shape[0]))
        shuffle(seg_inds)
        segments = segments[seg_inds, ...]
        labels = labels[seg_inds, ...]
        raise NotImplementedError
