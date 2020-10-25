"""
data generator for feeding data into pytorch models

Augmentations:
--------------
    - label smoothing (label, on the fly)
    - flip (signal, on the fly)
    - (re-)normalize to random mean and std (signal, on the fly)
    - baseline wander (signal, offline, combination of sinusoidal noise of several different frequencies, together with an optional Gaussian noise)
    - sinusoidal noise (signal, offline, done in baseline wander)
    - Gaussian noise (signal, offline, done in baseline wander)
    - stretch and compress (signal, offline)

References:
-----------
[1] Cai, Wenjie, and Danqin Hu. "QRS complex detection using novel deep learning neural networks." IEEE Access (2020).
[2] Tan, Jen Hong, et al. "Application of stacked convolutional and long short-term memory network for accurate identification of CAD ECG signals." Computers in biology and medicine 94 (2018): 19-26.
[3] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
"""
import os, sys
import json
from random import shuffle, randint, uniform
from copy import deepcopy
from functools import reduce
from itertools import product, repeat
from typing import Union, Optional, List, Tuple, Dict, Sequence, Set, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from scipy import signal as SS
from scipy.io import loadmat, savemat
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
    dict_to_str, mask_to_intervals, list_sum,
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
    __DEBUG__ = True
    __name__ = "CPSC2020"

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

        self.training = training
        split_res = self.reader.train_test_split_rec(
            test_rec_num=self.config.test_rec_num
        )
        self.__data_aug = self.training

        self.siglen = self.config.input_len  # alias, for simplicity

        # create directories if needed
        # preprocess_dir stores pre-processed signals
        self.preprocess_dir = os.path.join(config.db_dir, "preprocessed")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        # segments_dir for sliced segments
        self.segments_dir = os.path.join(config.db_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)
        # rpeaks_dir for detected r peaks, for optional use
        self.rpeaks_dir = os.path.join(config.db_dir, "rpeaks")
        os.makedirs(self.rpeaks_dir, exist_ok=True)

        if self.config.model_name.lower() == "crnn":
            self.segments_dirs = ED()
            for item in ["data", "ann"]:
                for rec in self.reader.all_records:
                    self.segments_dirs[item][rec] = os.path.join(self.segments_dir, item, rec)
                    os.makedirs(self.segments_dirs[item][rec], exist_ok=True)
            self.__all_segments = list_sum([
                os.listdir(self.segments_dirs.data[rec]) \
                    for rec in self.reader.all_records
            ])

            if self.training:
                self.segments = list_sum([
                    os.listdir(self.segments_dirs.data[rec]) \
                        for rec in split_res.train
                ])
            else:
                self.segments = list_sum([
                    os.listdir(self.segments_dirs.data[rec]) \
                        for rec in split_res.test
                ])
        else:
            raise NotImplementedError(f"data generator for model \042{self.config.model_name}\042 not implemented")


    def __getitem__(self, index:int) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        seg_data_fp = self._get_seg_data_path(self.segments[index])
        data = loadmat(seg_data_fp)["ecg"]
        seg_ann_fp = self._get_seg_ann_path(self.segments[index])
        ann = loadmat(seg_ann_fp)
        label = ann["label"]
        spb_indices = ann["SPB_indices"]
        pvc_indices = ann["PVC_indices"]
        raise NotImplementedError


    def __len__(self) -> int:
        """
        """
        return len(self.segments)


    def _get_seg_data_path(self, seg:str) -> str:
        """
        """
        rec = seg.split("_")[0].replace("S", "A")
        fp = os.path.join(self.segments_dir, "data", rec)
        return fp


    def _get_seg_ann_path(self, seg:str) -> str:
        """
        """
        rec = seg.split("_")[0].replace("S", "A")
        fp = os.path.join(self.segments_dir, "ann", rec)
        return fp


    def disable_data_augmentation(self) -> NoReturn:
        """
        """
        self.__data_aug = False


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
        config = deepcopy(PreprocCfg)
        config.preproc = preproc
        for rec in self.reader.all_records:
            self._preprocess_one_record(rec=rec, config=config)

    def _preprocess_one_record(self, rec:Union[int,str], config:dict) -> NoReturn:
        """ finished, checked,

        preprocesses the ecg data in advance for further use,
        offline for `self.persistence`

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        config: dict,
            configurations of preprocessing
        """
        # format save path
        save_fp = ED()
        rec_name = self.reader._get_rec_name(rec)
        suffix = self._get_rec_suffix(config.preproc)
        save_fp.data = os.path.join(self.preprocess_dir, f"{rec_name}-{suffix}{self.reader.rec_ext}")
        save_fp.rpeaks = os.path.join(self.rpeaks_dir, f"{rec_name}-{suffix}{self.reader.rec_ext}")
        if (not force_recompute) and os.path.isfile(save_fp.data) and os.path.isfile(save_fp.rpeaks):
            return
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
        """ finished, NOT checked,

        slice all records into segments of length `self.config.input_len`, i.e. `self.siglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters:
        -----------
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        """
        for rec in self.reader.all_records:
            self._slice_one_record(
                rec=rec,
                force_recompute=force_recompute
            )

    def _slice_one_record(self, rec:Union[int,str], force_recompute:bool=False) -> NoReturn:
        """ finished, NOT checked,

        slice one record into segments of length `self.config.input_len`, i.e. `self.siglen`,
        and perform data augmentations specified in `self.config`
        
        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        force_recompute: bool, default False,
            if True, recompute regardless of possible existing files
        """
        rec_name = self.reader._get_rec_name(rec)
        save_dirs = ED()
        save_dirs.data = self.segments_dirs.data[rec_name]
        save_dirs.ann = self.segments_dirs.ann[rec_name]
        os.makedirs(save_dirs.data, exist_ok=True)
        os.makedirs(save_dirs.data, exist_ok=True)
        if not force_recompute and 

        data = self.reader.load_data(rec, units="mV", keep_dim=False)
        ann = self.reader.load_ann(rec)
        border_dist = int(2 * self.config.fs)
        forward_len = self.siglen - self.config.overlap_len

        spb_mask = np.zeros((len(data),), dtype=int)
        pvc_mask = np.zeros((len(data),), dtype=int)
        spb_mask[ann["SPB_indices"]] = 1
        pvc_mask[ann["PVC_indices"]] = 1

        # generate initial segments with no overlap for non premature beats
        n_init_seg = len(data) // self.siglen
        segments = (data[:self.siglen*n_init_seg]).reshape((n_init_seg, self.siglen))
        labels = np.zeros((n_init_seg, self.n_classes))
        labels[..., self.config.class_map["N"]] = 1
        # for idx in range(n_init_seg):
        #     start_idx = idx * self.siglen
        #     end_idx = start_idx + self.siglen
        #     if spb_mask[start_idx:end_idx].any():
        #         labels[idx, self.config.class_map["S"]] = 1
        #         labels[idx, self.config.class_map["N"]] = 0
        #     if pvc_mask[start_idx:end_idx].any():
        #         labels[idx, self.config.class_map["V"]] = 1
        #         labels[idx, self.config.class_map["N"]] = 0
        # leave only non premature segments
        non_premature = np.logical_or(spb_mask, pvc_mask)[:self.siglen*n_init_seg]
        non_premature = non_premature.reshape((n_init_seg, self.siglen)).sum(axis=1)
        segments = segments[non_premature, ...]
        labels = labels[non_premature, ...]
        beat_ann = list(repeat(
            {"SPB_indices":np.array([],dtype=int), "PVC_indices":np.array([],dtype=int)},
            len(non_premature)
        ))

        # do data augmentation for premature beats
        # first locate all possible premature segments
        # mask for segment start indices
        premature_mask = np.zeros((len(data),), dtype=int)
        for idx in np.concatenate((ann["SPB_indices"], ann["PVC_indices"])):
            start_idx = max(0, idx-self.siglen+border_dist)
            end_idx = max(start_idx, min(idx-border_dist, len(data)-self.siglen))
            premature_mask[start_idx: end_idx] = 1
        # intervals for allowed start of augmented segments
        premature_intervals = mask_to_intervals(premature_mask, 1)
        for itv in premature_intervals:
            start_idx = itv[0]
            while start_idx < itv[1]:
                end_idx = start_idx + self.siglen

                # the segment of original signal, with no augmentation
                new_seg = data[start_idx:end_idx]
                seg_label = np.zeros((self.n_classes,))

                seg_spb_inds = np.where(spb_mask[start_idx: end_idx]==1)[0]
                seg_pvc_inds = np.where(pvc_mask[start_idx: end_idx]==1)[0]
                seg_beat_ann = {
                    "SPB_indices": seg_spb_inds,
                    "PVC_indices": seg_pvc_inds,
                }

                if len(seg_spb_inds) > 0:
                    seg_label[self.config.class_map["S"]] = 1
                if len(seg_pvc_inds) > 0:
                    seg_label[self.config.class_map["V"]] = 1
                seg_label = seg_label.reshape((1,-1))

                segments = np.append(segments, new_seg)
                labels = np.append(labels, seg_label.copy(), axis=0)
                beat_ann.append(seg_beat_ann.copy())

                seg_ampl = np.max(new_seg) - np.min(new_seg)
                # add baseline wander
                if self.config.bw:
                    for ar, (gm, gs) in product(self.config.bw_ampl_ratio, self.config.bw_gaussian):
                        bw_ampl = ar * seg_ampl
                        g_ampl = gm * seg_ampl
                        bw = gen_baseline_wander(
                            siglen=self.siglen,
                            fs=self.config.fs,
                            bw_fs=self.config.bw_fs,
                            amplitude=bw_ampl,
                            amplitude_mean=gm,
                            amplitude_std=gs,
                        )
                        aug_seg = (new_seg + bw).reshape((1,-1))
                        segments = np.append(segments, aug_seg, axis=0)
                        labels = np.append(labels, seg_label.copy(), axis=0)
                        beat_ann.append(seg_beat_ann.copy())
                # stretch and compress the signal
                if self.config.stretch_compress != 0:
                    for sign in [-1, 1]:
                        sc_ratio = self.config.stretch_compress
                        sc_ratio = 1 + (uniform(sc_ratio/4, sc_ratio) * sign) / 100
                        sc_len = int(round(sc_ratio * self.siglen))
                        aug_seg = data[start_idx: start_idx+sc_len]
                        aug_seg = SS.resample(x=aug_seg, num=self.siglen).reshape((1,-1))
                        sc_spb_inds = np.where(spb_mask[start_idx: start_idx+sc_len]==1)[0]
                        sc_pvc_inds = np.where(pvc_mask[start_idx: start_idx+sc_len]==1)[0]
                        sc_beat_ann = {
                            "SPB_indices": sc_spb_inds,
                            "PVC_indices": sc_pvc_inds,
                        }
                        sc_label = np.zeros((self.n_classes,))
                        if len(sc_spb_inds) > 0:
                            sc_label[self.config.class_map["S"]] = 1
                        if len(sc_pvc_inds) > 0:
                            sc_label[self.config.class_map["V"]] = 1
                        sc_label = sc_label.reshape((1,-1))
                        segments = np.append(segments, aug_seg, axis=0)
                        labels = np.append(labels, sc_label, axis=0)
                        beat_ann.append(sc_beat_ann)

                start_idx += forward_len

        # randomly shuffle the data and save into separate files
        seg_inds = list(range(segments.shape[0]))
        shuffle(seg_inds)
        for i, ind in enumerate(seg_inds):
            save_fp = ED()
            seg_name = f"{rec_name.replace('A', 'S')}_{i:07d}{self.reader.rec_ext}
            save_fp.data = os.path.join(self.record_dirs.data, seg_name)
            save_fp.ann = os.path.join(self.record_dirs.ann, seg_name)
            seg = segments[ind, ...]
            savemat(save_fp.data, {"ecg": seg}, format="5")
            seg_label = labels[ind, ...]
            seg_beat_ann = beat_ann[ind]
            save_ann_dict = seg_beat_ann.copy()
            save_ann_dict = save_ann_dict.update({"label": seg_label})
            savemat(save_fp.ann, save_ann_dict, format="5")

