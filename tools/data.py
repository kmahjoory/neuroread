
# Import libraries
import os, sys, glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

import mne

import matplotlib
import matplotlib.pyplot as plt
import time


def unfold_raw(raw, window_size=None, stride=None):
    """
    This function unfolds raw MNE object into a list of raw objects based on a sliding window.
    Args:
        raw: a raw MNE object cropped by rejecting bad segments.
    Returns:
        raw_unfolded: a raw MNE object unfolded by applying a sliding window.
    """
    if window_size is None:
        window_size = int(5 * raw.info['sfreq'])
    if stride is None:
        stride = window_size
    nchans = len(raw.ch_names)
    sig = torch.tensor(raw.get_data(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sig_unf = F.unfold(sig, (nchans, window_size), stride=stride , padding=0)
    sig_unf = sig_unf.permute(0, 2, 1).reshape(-1, sig_unf.shape[-1], nchans, window_size)
    return sig_unf


def rm_repeated_annotations(raw):
    """
    This functions taskes raw MNE obejct as input and removes repeated annotations.
    Repeated annotations are defined as annotations that are contained within another annotation.
    Args:
        raw: a raw MNE object cropped by rejecting bad segments.

     Returns:
        raw: a raw MNE object with repeated annotations removed.   
    """
    
    annots = raw.annotations.copy()
    annots_drop = []
    for k in annots:
        annots_drop.extend([k for kk in annots if (k['onset'] > kk['onset']) and (k['onset']+k['duration'] < kk['onset']+kk['duration']) ])

    annots_updated = [i for i in annots if i not in annots_drop]
    onsets = [i['onset'] for i in annots_updated]
    durations = [i['duration'] for i in annots_updated]
    descriptions = [i['description'] for i in annots_updated]
    print('Initial num of annots: %d  Num of removed annots: %d  Num of retained annots:  %d' % (len(annots), len(annots_drop), len(annots_updated)))
    print(f' New annots: {annots_updated}')
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions) ) 
    return raw