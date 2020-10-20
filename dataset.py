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
from utils import dict_to_str, mask_to_intervals

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


    def persistence(self) -> NoReturn:
        """ NOT finished, NOT checked,

        make the dataset persistent w.r.t. the ratios in `self.config`
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

    def _slice_data(self) -> NoReturn:
        """
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
            # generate initial segments with no overlap
            n_init_seg = len(data)//self.config.input_len
            segments = np.array(data[:self.config.input_len*n_init_seg]).reshape((n_init_seg, self.config.input_len))
            labels = np.zeros((n_init_seg, self.n_classes))
            labels[..., self.config.class_map["N"]] = 1
            for idx in range(n_init_seg):
                start_idx = idx * self.config.input_len
                end_idx = start_idx + self.config.input_len
                if spb_mask[start_idx:end_idx].any():
                    labels[idx, self.config.class_map["S"]] = 1
                    labels[idx, self.config.class_map["N"]] = 0
                if pvc_mask[start_idx:end_idx].any():
                    labels[idx, self.config.class_map["V"]] = 1
                    labels[idx, self.config.class_map["N"]] = 0

            # do data augmentation for premature beats
            aug_segments = np.array([], dtype=float)
            aug_labels = np.array([], dtype=int)
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
                    aug_segments = np.append(aug_segments, new_seg)
                    if spb_mask[start_idx:end_idx].any():
                        new_label[self.config.class_map["S"]] = 1
                    if pvc_mask[start_idx:end_idx].any():
                        new_label[self.config.class_map["V"]] = 1
                    aug_labels = np.append(aug_labels, new_label)
                    # TODO: perform data augmentation on such segments
                    if self.config.gaussian_std > 0:
                        guassian_noise = \
                            np.random.normal(0, self.config.gaussian_std, self.config.input_len)
                        aug_segments = np.append(aug_segments, new_seg + guassian_noise)
                        aug_labels = np.append(aug_labels, new_label)
                    if self.config.stretch_compress != 1:
                        pass
                    if self.config.flip:
                        pass
                    start_idx += forward_len

        raise NotImplementedError
