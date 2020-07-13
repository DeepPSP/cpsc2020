"""
"""
import os, sys
import argparse
import numpy as np

from signal_processing.ecg_preprocess import preprocess_signal, parallel_preprocess_signal
from signal_processing.ecg_features import compute_ecg_features
from models.load_model import load_model
from models.train_model_ml import train as train_ml
from models.train_model_dl import train as train_dl


def main(**kwargs):
    """
    """
    raise NotImplementedError


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="CPSC2020 extra configs",
    )
    ap.add_argument(
        "-m", "--mode",
        type=str, required=True,
        help="running mode, train or inference",
        dest="mode",
    )
    ap.add_argument(
        "-v", "--verbose",
        type=int, default=0,
        help="set verbosity",
        dest="verbose",
    )
    kwargs = vars(ap.parse_args())
    print("passed arguments:")
    print(repr(kwargs))
    main(**kwargs)
