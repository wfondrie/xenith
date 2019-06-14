"""
Defines the PsmDataset set class and the auxiliary functions needed to
easily construct one.
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data


class PsmDataset(torch.utils.data.Dataset):
    """
    Manage a collection of PSMs.

    Parameters
    ----------
    psm_files : str or tuple of str
        The files from which to load a set of PSMs. These should be in
        the xenith tab-delimited format.

    device : torch.device
        The device on which to put the tensors


    Attributes
    ----------
    metadata : pandas.DataFrame
        A dataframe containing all non-feature information about the
        PSMs.

    features : torch.FloatTensor
        A 2D tensor containing the features needed for training and
        prediction.

    
    """

