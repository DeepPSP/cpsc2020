"""
"""
import os
import random
import argparse
import math
from copy import deepcopy
from functools import reduce
import logging
from typing import Union, Optional, Any, List, Tuple, Dict, Sequence, NoReturn
from numbers import Real

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import multiprocessing as mp
from easydict import EasyDict as ED

from utils import CPSC_STATS, get_optimal_covering
from cfg import BaseCfg, PlotCfg


__all__ = [
    "CPSC2020Reader",
]


class CPSC2020Reader(object):
    """

    The 3rd China Physiological Signal Challenge 2020:
    Searching for Premature Ventricular Contraction (PVC) and Supraventricular Premature Beat (SPB) from Long-term ECGs

    ABOUT CPSC2019:
    ---------------
    1. training data consists of 10 single-lead ECG recordings collected from arrhythmia patients, each of the recording last for about 24 hours
    2. data and annotations are stored in v5 .mat files
    3. A02, A03, A08 are patient with atrial fibrillation
    4. sampling frequency = 400 Hz
    5. Detailed information:
        -------------------------------------------------------------------------
        rec   ?AF   Length(h)   # N beats   # V beats   # S beats   # Total beats
        A01   No	25.89       109,062     0           24          109,086
        A02   Yes	22.83       98,936      4,554       0           103,490
        A03   Yes	24.70       137,249     382         0           137,631
        A04   No	24.51       77,812      19,024      3,466       100,302
        A05   No	23.57       94,614  	1	        25	        94,640
        A06   No	24.59       77,621  	0	        6	        77,627
        A07   No	23.11	    73,325  	15,150	    3,481	    91,956
        A08   Yes	25.46	    115,518 	2,793	    0	        118,311
        A09   No	25.84	    88,229  	2	        1,462	    89,693
        A10   No	23.64	    72,821	    169	        9,071	    82,061
    6. challenging factors for accurate detection of SPB and PVC:
        amplitude variation; morphological variation; noise

    NOTE:
    -----
    1. the records can roughly be classified into 4 groups:
        N:  A01, A03, A05, A06
        V:  A02, A08
        S:  A09, A10
        VS: A04, A07
    2. as premature beats and atrial fibrillation can co-exists
    (via the following code, and data from CINC2020),
    the situation becomes more complicated.
    >>> from utils.scoring_aux_data import dx_cooccurrence_all
    >>> dx_cooccurrence_all.loc["AF", ["PAC","PVC","SVPB","VPB"]]
    ... PAC     20
    ... PVC     19
    ... SVPB     4
    ... VPB     20
    ... Name: AF, dtype: int64
    this could also be seen from this dataset, via the following code as an example:
    >>> from data_reader import CPSC2020Reader as CR
    >>> db_dir = '/media/cfs/wenhao71/data/CPSC2020/TrainingSet/'
    >>> dr = CR(db_dir)
    >>> rec = dr.all_records[1]
    >>> dr.plot(rec, sampfrom=0, sampto=4000, ticks_granularity=2)
    3. PVC and SPB can also co-exist, as illustrated via the following code (from CINC2020):
    >>> from utils.scoring_aux_data import dx_cooccurrence_all
    >>> dx_cooccurrence_all.loc[["PVC","VPB"], ["PAC","SVPB",]]
    ... 	PAC	SVPB
    ... PVC	14	1
    ... VPB	27	0
    and also from the following code:
    >>> for rec in dr.all_records:
    >>>     ann = dr.load_ann(rec)
    >>>     spb = ann["SPB_indices"]
    >>>     pvc = ann["PVC_indices"]
    >>>     if len(np.diff(spb)) > 0:
    >>>         print(f"{rec}: min dist among SPB = {np.min(np.diff(spb))}")
    >>>     if len(np.diff(pvc)) > 0:
    >>>         print(f"{rec}: min dist among PVC = {np.min(np.diff(pvc))}")
    >>>     diff = [s-p for s,p in product(spb, pvc)]
    >>>     if len(diff) > 0:
    >>>         print(f"{rec}: min dist between SPB and PVC = {np.min(np.abs(diff))}")
    ... A01: min dist among SPB = 630
    ... A02: min dist among SPB = 696
    ... A02: min dist among PVC = 87
    ... A02: min dist between SPB and PVC = 562
    ... A03: min dist among SPB = 7044
    ... A03: min dist among PVC = 151
    ... A03: min dist between SPB and PVC = 3750
    ... A04: min dist among SPB = 175
    ... A04: min dist among PVC = 156
    ... A04: min dist between SPB and PVC = 178
    ... A05: min dist among SPB = 182
    ... A05: min dist between SPB and PVC = 22320
    ... A06: min dist among SPB = 455158
    ... A07: min dist among SPB = 603
    ... A07: min dist among PVC = 153
    ... A07: min dist between SPB and PVC = 257
    ... A08: min dist among SPB = 2903029
    ... A08: min dist among PVC = 106
    ... A08: min dist between SPB and PVC = 350
    ... A09: min dist among SPB = 180
    ... A09: min dist among PVC = 7719290
    ... A09: min dist between SPB and PVC = 1271
    ... A10: min dist among SPB = 148
    ... A10: min dist among PVC = 708
    ... A10: min dist between SPB and PVC = 177

    ISSUES:
    -------
    1. currently, using `xqrs` as qrs detector,
       a lot more (more than 1000) rpeaks would be detected for A02, A07, A08,
       which might be caused by motion artefacts (or AF?);
       a lot less (more than 1000) rpeaks would be detected for A04.
       numeric details are as follows:
       ----------------------------------------------
       rec   ?AF    # beats by xqrs     # Total beats
       A01   No     109502              109,086
       A02   Yes    119562              103,490
       A03   Yes    135912              137,631
       A04   No     92746               100,302
       A05   No     94674               94,640
       A06   No     77955               77,627
       A07   No     98390               91,956
       A08   Yes    126908              118,311
       A09   No     89972               89,693
       A10   No     83509               82,061
    2. (fixed by an official update)
    A04 has duplicate 'PVC_indices' (13534856,27147621,35141190 all appear twice):
       before correction of `load_ann`:
       >>> from collections import Counter
       >>> db_dir = "/mnt/wenhao71/data/CPSC2020/TrainingSet/"
       >>> data_gen = CPSC2020Reader(db_dir=db_dir,working_dir=db_dir)
       >>> rec = 4
       >>> ann = data_gen.load_ann(rec)
       >>> Counter(ann['PVC_indices']).most_common()[:4]
       would produce [(13534856, 2), (27147621, 2), (35141190, 2), (848, 1)]
    3. when extracting morphological features using augmented rpeaks for A04,
       `RuntimeWarning: invalid value encountered in double_scalars` would raise
       for `R_value = (R_value - y_min) / (y_max - y_min)` and
       for `y_values[n] = (y_values[n] - y_min) / (y_max - y_min)`.
       this is caused by the 13882273-th sample, which is contained in 'PVC_indices',
       however, whether it is a PVC beat, or just motion artefact, is in doubt!

    TODO:
    -----
    1. use SNR to filter out too noisy segments?
    2. for ML, consider more features

    Usage:
    ------
    1. ecg arrhythmia (PVC, SPB) detection

    References:
    -----------
    [1] http://www.icbeb.org/CPSC2020.html
    [2] https://github.com/PIA-Group/BioSPPy
    """
    __name__ = "CPSC2020Reader"

    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=1, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_dir: str,
            directory where the database is stored
        working_dir: str, optional,
            working directory, to store intermediate files and log file
        verbose: int, default 2,
        """
        self.db_dir = db_dir
        self.working_dir = working_dir or os.getcwd()
        self.verbose = verbose

        self.fs = 400
        self.spacing = 1000/self.fs
        self.rec_ext = '.mat'
        self.ann_ext = '.mat'

        self.nb_records = 10
        self.all_records = ["A{0:02d}".format(i) for i in range(1,1+self.nb_records)]
        self.all_annotations = ["R{0:02d}".format(i) for i in range(1,1+self.nb_records)]
        self.all_references = self.all_annotations
        self.rec_dir = os.path.join(self.db_dir, "data")
        self.ann_dir = os.path.join(self.db_dir, "ref")
        self.data_dir = self.rec_dir
        self.ref_dir = self.ann_dir

        self.subgroups = ED({
            "N":  ["A01", "A03", "A05", "A06",],
            "V":  ["A02", "A08"],
            "S":  ["A09", "A10"],
            "VS": ["A04", "A07"],
        })

        self.df_stats = CPSC_STATS

        self.palette = {"spb": "yellow", "pvc": "red",}

        # a dict mapping the string annotations ('N', 'S', 'V') to digits (0, 1, 2)
        self.class_map = kwargs.get("class_map", BaseCfg.class_map)

        # TODO: add logger
    

    def load_data(self, rec:Union[int,str], units:str='mV', sampfrom:Optional[int]=None, sampto:Optional[int]=None, keep_dim:bool=True) -> np.ndarray:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        units: str, default 'mV',
            units of the output signal, can also be 'μV', with an alias of 'uV'
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        rec_name = self._get_rec_name(rec)
        rec_fp = os.path.join(self.data_dir, f"{rec_name}{self.rec_ext}")
        data = loadmat(rec_fp)['ecg']
        if units.lower() in ['uv', 'μv']:
            data = (1000 * data).astype(int)
        sf, st = (sampfrom or 0), (sampto or len(data))
        data = data[sf:st]
        if not keep_dim:
            data = data.flatten()
        return data


    def load_ann(self, rec:Union[int,str], sampfrom:Optional[int]=None, sampto:Optional[int]=None) -> Dict[str, np.ndarray]:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        
        Returns:
        --------
        ann: dict,
            with items (ndarray) "SPB_indices" and "PVC_indices",
            which record the indices of SPBs and PVCs
        """
        ann_name = self._get_ann_name(rec)
        ann_fp = os.path.join(self.ann_dir, ann_name + self.ann_ext)
        ann = loadmat(ann_fp)['ref']
        sf, st = (sampfrom or 0), (sampto or np.inf)
        spb_indices = ann['S_ref'][0,0].flatten().astype(int)
        # drop duplicates
        spb_indices = np.array(sorted(list(set(spb_indices))), dtype=int)
        spb_indices = spb_indices[np.where( (spb_indices>=sf) & (spb_indices<st) )[0]]
        pvc_indices = ann['V_ref'][0,0].flatten().astype(int)
        # drop duplicates
        pvc_indices = np.array(sorted(list(set(pvc_indices))), dtype=int)
        pvc_indices = pvc_indices[np.where( (pvc_indices>=sf) & (pvc_indices<st) )[0]]
        ann = {
            "SPB_indices": spb_indices,
            "PVC_indices": pvc_indices,
        }
        return ann


    def _get_ann_name(self, rec:Union[int,str]) -> str:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name

        Returns:
        --------
        ann_name: str,
            filename of the annotation file
        """
        if isinstance(rec, int):
            assert rec in range(1, self.nb_records+1), "rec should be in range(1,{})".format(self.nb_records+1)
            ann_name = self.all_annotations[rec-1]
        elif isinstance(rec, str):
            assert rec in self.all_annotations+self.all_records, "rec should be one of {} or one of {}".format(self.all_records, self.all_annotations)
            ann_name = rec.replace("A", "R")
        return ann_name


    def _get_rec_name(self, rec:Union[int,str]) -> str:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name

        Returns:
        --------
        rec_name: str,
            filename of the record
        """
        if isinstance(rec, int):
            assert rec in range(1, self.nb_records+1), "rec should be in range(1,{})".format(self.nb_records+1)
            rec_name = self.all_records[rec-1]
        elif isinstance(rec, str):
            assert rec in self.all_records, "rec should be one of {}".format(self.all_records)
            rec_name = rec
        return rec_name

    
    def train_test_split_rec(self, test_rec_num:int=2) -> Dict[str, List[str]]:
        """ finished, checked,

        split the records into train set and test set

        Parameters:
        -----------
        test_rec_num: int,
            number of records for the test set

        Returns:
        --------
        split_res: dict,
            with items `train`, `test`, both being list of record names
        """
        if test_rec_num == 1:
            test_records = random.sample(self.subgroups.VS, 1)
        elif test_rec_num == 2:
            test_records = random.sample(self.subgroups.VS, 1) + random.sample(self.subgroups.N, 1)
        elif test_rec_num == 3:
            test_records = random.sample(self.subgroups.VS, 1) + random.sample(self.subgroups.N, 2)
        elif test_rec_num == 4:
            test_records = []
            for k in self.subgroups.keys():
                test_records += random.sample(self.subgroups[k], 1)
        else:
            raise ValueError("test data ratio too high")
        train_records = [r for r in self.all_records if r not in test_records]
        
        split_res = ED({
            "train": train_records,
            "test": test_records,
        })
        
        return split_res


    def locate_premature_beats(self, rec:Union[int,str], premature_type:Optional[str]=None, window:int=10000, sampfrom:Optional[int]=None, sampto:Optional[int]=None) -> List[List[int]]:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        premature_type: str, optional,
            premature beat type, can be one of "SPB", "PVC"
        window: int, default 10000,
            window length of each premature beat
        sampfrom: int, optional,
            start index of the premature beats to locate
        sampto: int, optional,
            end index of the premature beats to locate

        Returns:
        --------
        premature_intervals: list,
            list of intervals of premature beats
        """
        ann = self.load_ann(rec)
        if premature_type:
            premature_inds = ann[f"{premature_type.upper()}_indices"]
        else:
            premature_inds = np.append(ann["SPB_indices"], ann["PVC_indices"])
            premature_inds = np.sort(premature_inds)
        try:  # premature_inds empty?
            sf, st = (sampfrom or 0), (sampto or premature_inds[-1]+1)
        except:
            premature_intervals = []
            return premature_intervals
        premature_inds = premature_inds[(sf < premature_inds) & (premature_inds < st)]
        tot_interval = [sf, st]
        premature_intervals, _ = get_optimal_covering(
            total_interval=tot_interval,
            to_cover=premature_inds,
            min_len=window*self.fs//1000,
            split_threshold=window*self.fs//1000,
            traceback=False,
        )
        return premature_intervals


    def _auto_infer_units(self, sig:np.ndarray, sig_type:str="ECG") -> str:
        """ finished, checked,

        automatically infer the units of `sig`,
        under the assumption that `sig` not being raw signal, with baseline removed

        Parameters:
        -----------
        sig: ndarray,
            the signal to infer its units
        sig_type: str, default "ECG", case insensitive,
            type of the signal

        Returns:
        --------
        units: str,
            units of `sig`, 'μV' or 'mV'
        """
        if sig_type.lower() == "ecg":
            _MAX_mV = 20  # 20mV, seldom an ECG device has range larger than this value
            max_val = np.max(np.abs(sig))
            if max_val > _MAX_mV:
                units = 'μV'
            else:
                units = 'mV'
        else:
            raise NotImplementedError(f"not implemented for {sig_type}")
        return units

    
    def plot(self, rec:Union[int,str], data:Optional[np.ndarray]=None, ann:Optional[Dict[str, np.ndarray]]=None, ticks_granularity:int=0, sampfrom:Optional[int]=None, sampto:Optional[int]=None, rpeak_inds:Optional[Union[Sequence[int],np.ndarray]]=None) -> NoReturn:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        data: ndarray, optional,
            ecg signal to plot,
            if given, data of `rec` will not be used,
            this is useful when plotting filtered data
        ann: dict, optional,
            annotations for `data`,
            "SPB_indices", "PVC_indices", each of ndarray values,
            ignored if `data` is None
        ticks_granularity: int, default 0,
            the granularity to plot axis ticks, the higher the more,
            0 (no ticks) --> 1 (major ticks) --> 2 (major + minor ticks)
        sampfrom: int, optional,
            start index of the data to plot
        sampto: int, optional,
            end index of the data to plot
        rpeak_inds: array_like, optional,
            indices of R peaks,
            if `data` is None, then indices should be the absolute indices in the record
        """
        if 'plt' not in dir():
            import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        patches = {}

        if data is None:
            _data = self.load_data(
                rec, units="μV", sampfrom=sampfrom, sampto=sampto, keep_dim=False
            )
        else:
            units = self._auto_infer_units(data)
            if units == "mV":
                _data = data * 1000
            elif units == "μV":
                _data = data.copy()

        if ann is None or data is None:
            ann = self.load_ann(rec, sampfrom=sampfrom, sampto=sampto)
        sf, st = (sampfrom or 0), (sampto or len(_data))
        spb_indices = ann["SPB_indices"]
        pvc_indices = ann["PVC_indices"]
        spb_indices = spb_indices - sf
        pvc_indices = pvc_indices - sf

        if rpeak_inds is not None:
            if data is not None:
                rpeak_secs = np.array(rpeak_inds) / self.fs
            else:
                rpeak_secs = np.array(rpeak_inds)
                rpeak_secs = rpeak_secs[np.where( (rpeak_secs>=sf) & (rpeak_secs<st))[0]]
                rpeak_secs = (rpeak_secs - sf) / self.fs

        line_len = self.fs * 25  # 25 seconds
        nb_lines = math.ceil(len(_data)/line_len)

        for idx in range(nb_lines):
            seg = _data[idx*line_len: (idx+1)*line_len]
            secs = (np.arange(len(seg)) + idx*line_len) / self.fs
            fig_sz_w = int(round(4.8 * len(seg) / self.fs))
            y_range = np.max(np.abs(seg)) + 100
            fig_sz_h = 6 * y_range / 1500
            fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
            ax.plot(secs, seg, c='black')
            ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
            if ticks_granularity >= 1:
                ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
                ax.yaxis.set_major_locator(plt.MultipleLocator(500))
                ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
            if ticks_granularity >= 2:
                ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            seg_spb = np.where( (spb_indices>=idx*line_len) & (spb_indices<(idx+1)*line_len) )[0]
            # print(f"spb_indices = {spb_indices}, seg_spb = {seg_spb}")
            if len(seg_spb) > 0:
                seg_spb = spb_indices[seg_spb] / self.fs
                patches["SPB"] = mpatches.Patch(color=self.palette["spb"], label="SPB")
            seg_pvc = np.where( (pvc_indices>=idx*line_len) & (pvc_indices<(idx+1)*line_len) )[0]
            # print(f"pvc_indices = {pvc_indices}, seg_pvc = {seg_pvc}")
            if len(seg_pvc) > 0:
                seg_pvc = pvc_indices[seg_pvc] / self.fs
                patches["PVC"] = mpatches.Patch(color=self.palette["pvc"], label="PVC")
            for t in seg_spb:
                ax.axvspan(
                    max(secs[0], t-BaseCfg.bias_thr/self.fs), min(secs[-1], t+BaseCfg.bias_thr/self.fs),
                    color=self.palette["spb"], alpha=0.3
                )
                ax.axvspan(
                    max(secs[0], t-PlotCfg.winL), min(secs[-1], t+PlotCfg.winR),
                    color=self.palette["spb"], alpha=0.9
                )
            for t in seg_pvc:
                ax.axvspan(
                    max(secs[0], t-BaseCfg.bias_thr/self.fs), min(secs[-1], t+BaseCfg.bias_thr/self.fs),
                    color=self.palette["pvc"], alpha=0.3
                )
                ax.axvspan(
                    max(secs[0], t-PlotCfg.winL), min(secs[-1], t+PlotCfg.winR),
                    color=self.palette["pvc"], alpha=0.9
                )
            if len(patches) > 0:
                ax.legend(
                    handles=[v for _,v in patches.items()],
                    loc="lower left",
                    prop={"size": 16}
                )
            if rpeak_inds is not None:
                seg_rpeak_secs = \
                    rpeak_secs[np.where( (rpeak_secs>=secs[0]) & (rpeak_secs<secs[-1]))[0]]
                for r in seg_rpeak_secs:
                    ax.axvspan(r-0.01, r+0.01, color='green', alpha=0.7)
            ax.set_xlim(secs[0], secs[-1])
            ax.set_ylim(-y_range, y_range)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Voltage [μV]')
            plt.show()



if __name__ == "__main__":
    from .utils import dict_to_str, str2bool
    ap = argparse.ArgumentParser(
        description="preprocess CPSC2020 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "-d", "--db-dir",
        type=str, required=True,
        help="directory where the database is stored",
        dest="db_dir",
    )
    ap.add_argument(
        "-w", "--working-dir",
        type=str, default=None,
        help="working directory",
        dest="working_dir",
    )
    ap.add_argument(
        "-p", "--preproc",
        type=str, default="baseline,bandpass",
        help="preprocesses to perform, separated by ','",
        dest="preproc",
    )
    ap.add_argument(
        "-r", "--rec",
        type=str, default=None,
        help="records (name or numbering) to perform preprocesses, separated by ','; if not set, all records will be preprocessed",
        dest="records",
    )
    ap.add_argument(
        "-a", "--augment",
        type=str2bool, default=True,
        help="whether or not using annotations to augment the rpeaks detected by algorithm",
        dest="augment",
    )
    ap.add_argument(
        "-v", "--verbose",
        type=int, default=2,
        help="verbosity",
        dest="verbose",
    )
    # TODO: add more args

    kwargs = vars(ap.parse_args())
    print("passed arguments:")
    print(f"{dict_to_str(kwargs)}")

    # data_gen = CPSC2020Reader(db_dir="/mnt/wenhao71/data/CPSC2020/TrainingSet/")
    data_gen = CPSC2020Reader(
        db_dir=kwargs.get("db_dir"),
        working_dir=kwargs.get("working_dir"),
        verbose=kwargs.get("verbose"),
    )
