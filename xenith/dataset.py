"""
Defines the PsmDataset set class and the auxiliary functions needed to
easily construct one.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import xenith.fdr

# Classes ---------------------------------------------------------------------
class _PsmDataset(torch.utils.data.Dataset):
    """
    A Dataset class for using a XenithDataset with pytorch models.

    While the XenithDataset stores the features, metadata, and
    predictions in pandas.DataFrames, the _PsmDataset class stores
    the normalized features and targets each as a torch.FloatTensor().

    Parameters
    ----------
    xenith_dataset : xenith.XenithDataset
        A XenithDataset object to get the features and target
        information from.

    feat_mean : pd.Series
    feat_stdev : pd.Series
        pandas Series containing the mean and standard deviation used for the
        normalization of each feature. If `None`, the these are calculated
        from the XenithDataset.features.

    normalize : bool
        Should features be normalized?

    Attributes
    ----------
    features : torch.FloatTensor
        A 2D tensor containing the normalized features. Dimension 0 is
        each a PSM and dimensions 1 are features

    target : torch.FloatTensor
        A 1D tensor indicating whether the PSM is a target hit or not.
        target-target hits are 1, whereas target-decoy and decoy-decoy
        hits are 0.

    feat_mean : pd.Series
    feat_stdev : pd.Series
        pandas Series containing the mean and standard deviation that were
        used for the normalization of each feature, if normalization was
        performed.
    """
    def __init__(self, xenith_dataset, feat_mean, feat_stdev, normalize):
        """Instantiate a _PsmDataset object"""
        norm_feat = _process_features(xenith_dataset.features,
                                      feat_mean=feat_mean,
                                      feat_stdev=feat_stdev,
                                      normalize=normalize)

        self.feat_mean = norm_feat[1]
        self.feat_stdev = norm_feat[2]
        self.features = torch.FloatTensor(norm_feat[0].values)
        self.target = xenith_dataset.metadata.numtarget.values == 2
        self.target = torch.FloatTensor(self.target.astype(int))

    def __len__(self):
        """Get the total number of samples"""
        return len(self.target)

    def __getitem__(self, idx):
        """Generate one sample of data"""
        return [self.target[idx], self.features[idx, :]]


class XenithDataset():
    """
    Manage a collection of PSMs.

    Parameters
    ----------
    psm_files : str or tuple of str
        The files from which to load a set of PSMs. These should be in
        the xenith tab-delimited format.

    additional_metadata : tuple of str
        Additional columns to be considered as metadata. The columns
        specified here will not be included as features.

    Attributes
    ----------
    metadata : pandas.DataFrame
        A dataframe containing all non-feature information about the
        PSMs.

    features : pandas.DataFrame
        A dataframe containing the raw values of features for model
        training and prediction.

    predictions : pandas.DataFrame
        A dataframe containing new scores for each PSM. New predictions
        (such as from using different models) will be added as a new
        column. If no predictions have been made, this is an empty
        dataframe.
    """
    def __init__(self, psm_files: Tuple[str], additional_metadata=None):
        """Initialize a PsmDataset"""
        meta_cols = ["psmid", "numtarget", "scannr", "peptidea", "peptideb",
                     "peptidelinksitea", "peptidelinksiteb",
                     "proteinlinksitea", "proteinlinksiteb",
                     "proteina", "proteinb", "fileidx"]

        if additional_metadata is not None:
            if not isinstance(additional_metadata, list):
                additional_metadata = [str(additional_metadata)]

            additional_metadata = [m.lower() for m in additional_metadata]
            meta_cols = meta_cols + additional_metadata

        psms = _parse_psms(psm_files, meta_cols, additional_metadata)
        self.metadata = psms.loc[:, meta_cols]
        self.features = psms.drop(columns=meta_cols)
        self.predictions = pd.DataFrame()

    def estimate_qvalues(self, metric: str = "XenithScore",
                         desc: bool = True) -> Tuple[pd.DataFrame]:
        """
        Estimate q-values at the PSM, and cross-link levels.

        At each level, the false discovery rate (FDR) is estimated using
        target-decoy competition. For PSMs, ties from the same scan are
        broken randomly. Peptide aggregation is performed using the top
        scoring PSM for a peptide. FDR at the cross-link level is
        estimated using only unambiguous peptides; That is, peptides
        that correspond to a single protein and linked residue for
        each peptide.

        Parameters
        ----------
        metric : str
            The metric by which to rank PSMs. This can either be a model
            prediction or any feature. This is case-sensitive.

        level : str
            The level at which to estimate q-values. Can be one of
            'psm', 'peptide', or 'cross-link'.

        desc : bool
            Does a higher value of metric indicate a better PSM?

        Returns
        -------
        Tuple[pandas.DataFrame]
            A DataFrame with the q-values at the PSM, and cross-link
            level, respectively.
        """
        in_pred = metric in self.predictions.columns.tolist()
        in_feat = metric in self.features.columns.tolist()

        if in_pred and in_feat:
            res_df = self.predictions.loc[:, metric]
            logging.warning("'%s' was found in both the predictions and "
                            "features of the XenithDataset. Using the "
                            "predictions.", metric)
        elif in_pred and not in_feat:
            res_df = self.predictions.loc[:, metric]
        elif in_feat and not in_pred:
            res_df = self.features.loc[:, metric]
        else:
            raise ValueError(f"'{metric}' was not found in the predictions or "
                             "features of the XenithDataset.")

        res_df = pd.concat([self.metadata, res_df], axis=1)

        # Generate keys for grouping
        prot_site1 = (res_df.proteina + "_"
                      + res_df.proteinlinksitea.astype(str))
        prot_site2 = (res_df.proteinb + "_"
                      + res_df.proteinlinksiteb.astype(str))

        res_df["residue_key"] = ["--".join(sorted(x))
                                 for x in zip(prot_site1, prot_site2)]

        # randomize the df, so that ties are broken randomly
        res_df = res_df.sample(frac=1).reset_index(drop=True)

        psm_cols = ["fileidx", "scannr"]

        # PSM FDR -------------------------------------------------------------
        psm_idx = res_df.groupby(psm_cols)[metric].idxmax()
        psms = res_df.loc[psm_idx, :]

        # Cross-Link FDR ------------------------------------------------------
        link_idx = psms.groupby("residue_key")[metric].idxmax()
        links = psms.loc[link_idx, :]
        links = links.loc[~links.residue_key.str.contains(";")]

        # Estimate q-values ----------------------------------------------------
        out_list = []
        for dat in (psms, links):
            dat["q-values"] = xenith.fdr.qvalues(dat.numtarget.values,
                                                 dat[metric].values,
                                                 desc=desc)
            dat.sort_values(metric, ascending=(not desc), inplace=True)
            dat.reset_index(drop=True, inplace=True)
            out_list.append(_format_output(dat, metric))

        return out_list


# Functions -------------------------------------------------------------------
def load_psms(psm_files: Tuple[str], additional_metadata: Tuple[str] = None)\
    -> XenithDataset:
    """
    Load a collection of peptide-spectrum matches (PSMs).

    Reads a collection of PSMs from a file in the xenith tab-delimited
    format. By default, the required fields are considered metadata
    whereas all other fields are considered features.

    Parameters
    ----------
    psm_files : str or tuple of str
        The files from which to load a set of PSMs. These should be in
        the xenith tab-delimited format.

    additional_metadata : tuple of str
        Additional columns to be considered metadata. The columns
        specified here will not be included as features.

    Returns
    -------
    xenith.dataset.XenithDataset
        A XenithDataset object containing the PSMs.
    """
    return XenithDataset(psm_files=psm_files,
                         additional_metadata=additional_metadata)


# Utility Functions -----------------------------------------------------------
def _format_output(out_df, metric):
    """
    Format the output dataframe from estimate_qvalues()

    Parameters
    ----------
    out_df : pandas.DataFrame
        The dataframe with q-values and the other output columns.
    """
    order = ["fileidx", "psmid", "numtarget", "scannr", metric,
             "q-values", "peptidea", "peptideb", "peptidelinksitea",
             "peptidelinksiteb", "proteinlinksitea", "proteinlinksiteb",
             "proteina", "proteinb"]

    out_df = out_df.loc[:, order]

    out_df = out_df.rename(columns={"fileidx": "FileIdx",
                                    "psmid": "PsmId",
                                    "numtarget": "NumTarget",
                                    "scannr": "ScanNr",
                                    "peptidea": "PeptideA",
                                    "peptideb": "PeptideB",
                                    "peptidelinksitea": "PeptideLinkSiteA",
                                    "peptidelinksiteb": "PeptideLinkSiteB",
                                    "proteinlinksitea": "ProteinLinkSiteA",
                                    "proteinlinksiteb": "ProteinLinkSiteB",
                                    "proteina": "ProteinA",
                                    "proteinb": "ProteinB"})

    return out_df


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
    if not all(np.issubdtype(col.dtype, np.number)
               for _, col in feat_df.iteritems()):
        raise ValueError("All feature columns must be numeric.")

    if feat_mean is None:
        feat_mean = feat_df.mean()
    if feat_stdev is None:
        feat_stdev = feat_df.std(ddof=0)

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


def _parse_psms(psm_files, meta_cols, more_meta_cols):
    """
    Parse the PSM list and throw appropriate errors.

    Parameters
    ----------
    psm_files : list
        A list of files in xenith tab-delimited format.

    meta_cols : list
        A list of the expected metadata columns.

    more_meta_cols : list
        A list of additional metadata columns.

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

        logging.info("Assigning fileidx %s to %s.", idx, psm_file)
        psms["fileidx"] = idx

        if not idx:
            feat_set = set(psms.columns)

        if (more_meta_cols is not None and
            not set(more_meta_cols) <= set(psms.columns)):
            raise RuntimeError(f"{psm_file} does not contain the specified "
                               "additional metadata columns.")

        if not set(meta_cols) <= set(psms.columns):
            raise RuntimeError(f"{psm_file} either does not contain the required "
                               "columns for xenith.")

        if not feat_set == set(psms.columns):
            raise RuntimeError(f"Features in {psm_file} do not match the"
                               "features from the previous files.")

        psm_list.append(psms)

    psms = pd.concat(psm_list)
    psms = psms.reset_index(drop=True)
    return psms
