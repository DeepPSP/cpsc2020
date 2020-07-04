"""
"""
import os
from typing import Union, Optional, Any, List, Dict, NoReturn
from numbers import Real
import numpy as np
from scipy.io import loadmat, savemat
from easydict import EasyDict as ED

import misc
from signal_processing.ecg_preprocess import parallel_preprocess_signal
from signal_processing.ecg_features import compute_ecg_features


class CPSC2020(object):
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

    ISSUES:
    -------

    Usage:
    ------
    1. ecg arrhythmia (PVC, SPB) detection

    References:
    -----------
    [1] http://www.icbeb.org/CPSC2020.html
    [2] https://github.com/mondejar/ecg-classification
    """
    def __init__(self, db_dir:str, working_dir:Optional[str]=None, verbose:int=2, **kwargs):
        """ finished, to be improved,

        Parameters:
        -----------
        db_dir: str,
            storage path of the database
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

        self.palette = {"spb": "black", "pvc": "red",}

        self.allowed_preprocess = ['baseline', 'bandpass']
        self.preprocess_dir = os.path.join(self.db_dir, "preprocessed")
        os.makedirs(self.preprocess_dir, exist_ok=True)
        self.rpeaks_dir = os.path.join(self.db_dir, "rpeaks")
        os.makedirs(self.rpeaks_dir, exist_ok=True)
        self.feature_dir = os.path.join(self.db_dir, "features")
        os.makedirs(self.feature_dir, exist_ok=True)
    

    def load_data(self, rec:Union[int,str], sampfrom:Optional[int]=None, sampto:Optional[int]=None, keep_dim:bool=True, preprocess:Optional[List[str]]=None) -> np.ndarray:
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
        keep_dim: bool, default True,
            whether or not to flatten the data of shape (n,1)
        
        Returns:
        --------
        data: ndarray,
            the ecg data
        """
        preprocess = preprocess or [item.lower() for item in preprocess]
        assert all([item in self.allowed_preprocess for item in preprocess])
        rec_name = self._get_rec_name(rec)
        if preprocess:
            rec_name = f"{rec_name}-{self._get_rec_suffix(preprocess)}"
            rec_fp = os.path.join(self.preprocess_dir, f"{rec_name}{self.rec_ext}")
        else:
            rec_fp = os.path.join(self.data_dir, f"{rec_name}{self.rec_ext}")
        data = (1000 * loadmat(rec_fp)['ecg']).astype(int)
        sf, st = (sampfrom or 0), (sampto or len(data))
        data = data[sf:st]
        if not keep_dim:
            data = data.flatten()
        return data


    def preprocess_data(self, rec:Union[int,str], preprocess:List[str]) -> NoReturn:
        """

        preprocess the ecg data in advance for further use

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        preprocess: list of str,
            type of preprocess to perform, should be sublist of `self.allowed_preprocess`
        """
        preprocess = preprocess or [item.lower() for item in preprocess]
        assert preprocess and all([item in self.allowed_preprocess for item in preprocess])
        save_fp = ED()
        save_fp.data = os.path.join(self.preprocess_dir, f"{rec_name}-{self._get_rec_suffix(preprocess)}{self.rec_ext}")
        save_fp.rpeaks = os.path.join(self.rpeaks_dir, f"{rec_name}-{self._get_rec_suffix(preprocess)}{self.rec_ext}")
        config = ED()
        config.remove_baseline = ('baseline' in preprocess)
        config.filter_signal = ('bandpass' in preprocess)
        pps = parallel_preprocess_signal(self.load_data(rec), fs=self.fs, config=config)
        # save mat, keep in accordance with original mat files
        savemat(save_fp.data, {'ecg': np.atleast_2d(pps['filtered_ecg']).T}, format='5')
        savemat(save_fp.rpeaks, {'rpeaks': np.atleast_2d(pps['rpeaks']).T}, format='5')


    def load_ann(self, rec:Union[int,str], sampfrom:Optional[int]=None, sampto:Optional[int]=None) -> Dict[str, np.ndarray]:
        """ finished, checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        
        Returns:
        --------
        ann: dict,
            with items "SPB_indices" and "PVC_indices", which record the indices of SPBs and PVCs
        """
        if isinstance(rec, int):
            assert rec in range(1, self.nb_records+1), "rec should be in range(1,{})".format(self.nb_records+1)
            ann_name = self.all_annotations[rec-1]
        elif isinstance(rec, str):
            assert rec in self.all_annotations+self.all_records, "rec should be one of {} or one of {}".format(self.all_records, self.all_annotations)
            ann_name = rec.replace("A", "R")
        ann_fp = os.path.join(self.ann_dir, ann_name + self.ann_ext)
        ann = loadmat(ann_fp)['ref']
        sf, st = (sampfrom or 0), (sampto or np.inf)
        ann = {
            "SPB_indices": [p for p in ann['S_ref'][0,0].flatten() if sf<=p<st],
            "PVC_indices": [p for p in ann['V_ref'][0,0].flatten() if sf<=p<st],
        }
        return ann


    def _get_ann_name(self, rec:Union[int,str]) -> str:
        """

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
        """

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


    def _get_rec_suffix(self, preprocess:List[str]) -> str:
        """

        Parameters:
        -----------
        preprocess: list of str,
            type of preprocess to perform, should be sublist of `self.allowed_preprocess`

        Returns:
        --------
        suffix: str,
            suffix of the filename of the preprocessed ecg signal
        """
        suffix = '-'.join(sorted([item.lower() for item in preprocess]))
        return suffix

    
    def plot(self, rec:Union[int,str], sampfrom:Optional[int]=None, sampto:Optional[int]=None, ectopic_beats_only:bool=False, **kwargs) -> NoReturn:
        """ not finished, not checked,

        Parameters:
        -----------
        rec: int or str,
            number of the record, NOTE that rec_no starts from 1,
            or the record name
        sampfrom: int, optional,
            start index of the data to be loaded
        sampto: int, optional,
            end index of the data to be loaded
        ectopic_beats_only: bool, default False,
            whether or not onpy plot the neighborhoods of the ectopic beats
        """
        data = self.load_data(rec, sampfrom=sampfrom, sampto=sampto, keep_dim=False)
        ann = self.load_ann(rec, sampfrom=sampfrom, sampto=sampto)
        sf, st = (sampfrom or 0), (sampto or len(data))
        if ectopic_beats_only:
            ectopic_beat_indices = sorted(ann["SPB_indices"] + ann["PVC_indices"])
            tot_interval = [sf, st]
            covering, tb = misc.get_optimal_covering(
                total_interval=tot_interval,
                to_cover=ectopic_beat_indices,
                min_len=3*self.freq,
                split_threshold=3*self.freq,
                traceback=True,
                verbose=self.verbose,
            )
        # TODO: finishe plot
        raise NotImplementedError

    
    def train_test_split(self, test_rec_num:int=2) -> dict:
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
