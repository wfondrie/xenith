"""
Defines the PsmDataset set class and the auxiliary functions needed to
easily construct one.
"""
import numpy as np
import pandas as pd
import torch
import torch.utils.data

# Classes ---------------------------------------------------------------------
class PsmDataset(torch.utils.data.Dataset):
    """
    Manage a collection of PSMs.

    Parameters
    ----------
    psm_files : str or tuple of str
        The files from which to load a set of PSMs. These should be in
        the xenith tab-delimited format.

    feat_mean : pd.DataFrame
    feat_stdev : pd.DataFrame
        Wide dataframes containing the mean and standard deviation of
        each feature to use for normalization. If `None`, these are
        calculated on the parsed data. For prediction, these should
        be the respective values from the training set.

    normalize : bool
        Should the features be standard deviation normalized? In the
        case of a model from Percolator, this should be `False`, because
        the raw weights will be used.

    addtional_metadata : tuple of str
        Additional columns to be considered metadata. This can be useful
        for removing specific features.

    device : torch.device
        The device on which to put the tensors

    Attributes
    ----------
    metadata : pandas.DataFrame
        A dataframe containing all non-feature information about the
        PSMs.

    feature_names : list
        A list of the parsed feature names.

    features : torch.FloatTensor
        A 2D tensor containing the features needed for
        training and prediction.

    target : torch.FloatTensor
        A 1D tensor indicating whether a PSM is a decoy or not. Decoys
        are defined as PSMs where one or more of the cross-linked
        peptides are a decoy sequence. `1` indicates target, `0`
        indicates decoy.

    metrics : pandas.DataFrame
        The model predictions. If no predictions have been made,
        this is `None`.

    feat_mean : pd.DataFrame
    feat_stdev : pd.DataFrame
        Wide dataframes containing the mean and standard deviation
        used for the normalization of each feature.
    """
    def __init__(self, psm_files, device, feat_mean=None, feat_stdev=None,
                 normalize=True, additional_metadata=None):
        """Initialize a PsmDataset"""
        meta_cols = ["specid", "numtarget", "scannr", "peptidea", "peptideb",
                     "linksitea", "linksiteb", "proteina", "proteinb",
                     "fileidx"]

        if additional_metadata is not None:
            meta_cols = meta_cols + additional_metadata

        psms = _parse_psms(psm_files, meta_cols)
        self.metadata = psms.loc[:, meta_cols]

        # Process features
        feat_df = psms.drop(columns=meta_cols)
        norm_feat = _process_features(feat_df, feat_mean, feat_stdev,
                                      normalize)

        self.feature_names = feat_df.columns.tolist()
        self.feat_mean = norm_feat[1]
        self.feat_stdev = norm_feat[2]

        self.features = torch.FloatTensor(norm_feat[0].values).to(device)
        self.target = torch.FloatTensor(self.metadata.numtarget == 2).to(device)
        self.metrics = pd.DataFrame()
        self._feat_df = feat_df

    def __len__(self):
        """Get the total number of sample"""
        return len(self.target)

    def __getitem__(self, idx):
        """Generate one sample of data"""
        return [self.target[idx], self.features[idx, :]]

    def add_metric(self, name, value=None):
        """
        Add a scoring metric

        Parameters
        ----------
        name : str
            Name of the new metric.

        value : array-like or None
            The metric values, such as the predictions from a
            XenithModel. If `None`, a feature that matches
            the `name` argument will be used.
        """
        if value is None:
            value = self._feat_df[name]

        self.metrics[name] = value

    def estimate_qvalues(self, level: str) -> "pandas.DataFrame":
        """
        Estimate q-values at the PSM, cross-link, and peptide levels.

        Parameters
        ----------
        level : str
            The level at which to estimate q-values. Can be one of
            'psm', 'cross-link', or 'peptide'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the q-values at the requested level.
        """
        res_df = pd.concat(self.metadata, self.metrics)

        return res_df


# Functions -------------------------------------------------------------------
def _process_features(feat_df, feat_mean, feat_stdev, normalize):
    """
    Process a dataframe of features.

    This function normalizes features and verifies that the `feat_mean`
    and `feat_stdev` have the same features as feat_df.

    Parameters
    ----------
    feat_df : pandas.DataFrame
        A dataframe containing only the features for training and
        prediction.

    feat_mean : pd.Series
    feat_stdev : pd.Series
        Series containing the mean and standard deviation of
        each feature to use for normalization. If `None`, these are
        calculated on the parsed data. For prediction, these should
        be the respective values from the training set.

    normalize : bool
        Should the features be normalized?

    Returns
    -------
    tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        A tuple of dataframes containing the normalized features, the
        employed feat_mean, and the employed feat_stdev, in order.
    """
    if feat_mean is None:
        feat_mean = feat_df.mean(numeric_only=True)
    if feat_stdev is None:
        feat_stdev = feat_df.std(ddof=0, numeric_only=True)

    feat_set = set(feat_df.columns)
    feat_mean_set = set(feat_mean.index)
    feat_stdev_set = set(feat_stdev.index)

    if feat_mean_set != feat_stdev_set:
        # This one should never happen with the public API.
        raise RuntimeError("Features for the normalization parameters "
                           "do not match.")

    if feat_set != feat_mean_set:
        raise RuntimeError("Model features do not match the dataset.")

    # Align features
    feat_mean = feat_mean.loc[feat_df.columns]
    feat_stdev = feat_stdev.loc[feat_df.columns]

    eps = np.finfo(np.float).eps
    if normalize:
        feat_df = (feat_df - feat_mean.values) / (feat_stdev.values + eps)

    return (feat_df, feat_mean, feat_stdev)


def _parse_psms(psm_files, meta_cols):
    """
    Parse the PSM list and throw appropriate errors.

    Parameters
    ----------
    psm_files : list
        A list of files in xenith tab-delimited format.

    meta_cols : list
        A list of the expected metadata columns.

    Returns
    -------
    pandas.DataFrame
        A dataframe containing a list of psms
    """
    if isinstance(psm_files, str):
        psm_files = (psm_files,)

    psm_list = []
    for idx, psm_file in enumerate(psm_files):
        psms = pd.read_csv(psm_file, sep="\t")
        psms.columns = psms.columns.str.lower()
        psms["fileidx"] = idx

        if not idx:
            feat_set = set(psms.columns)

        if not set(meta_cols) <= set(psms.columns):
            raise RuntimeError(f"{psm_file} does not contain the expected "
                               "columns for xenith. See ? for details.")

        if not feat_set == set(psms.columns):
            raise RuntimeError(f"Features in {psm_file} do not match the"
                               "features from the previous files.")

        psm_list.append(psms)

    return pd.concat(psm_list)
