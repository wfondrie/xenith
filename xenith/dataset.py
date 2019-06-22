"""
Defines the PsmDataset set class and the auxiliary functions needed to
easily construct one.
"""
import logging
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import xenith.fdr

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

    prediction : pandas.DataFrame
        The model predictions. If no predictions have been made,
        this is an empty dataframe.

    feat_mean : pd.Series
    feat_stdev : pd.Series
        pandas Series containing the mean and standard deviation used for the
        normalization of each feature.
    """
    def __init__(self, psm_files, feat_mean=None, feat_stdev=None,
                 normalize=True, additional_metadata=None):
        """Initialize a PsmDataset"""
        meta_cols = ["psmid", "numtarget", "scannr", "peptidea", "peptideb",
                     "peptidelinksitea", "peptidelinksiteb",
                     "proteinlinksitea", "proteinlinksiteb",
                     "proteina", "proteinb", "fileidx"]

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

        self.features = torch.FloatTensor(norm_feat[0].values)
        self.target = self.metadata.numtarget.values == 2
        self.target = torch.FloatTensor(self.target.astype(int))
        self.prediction = pd.DataFrame()
        self._feat_df = feat_df

    def __len__(self):
        """Get the total number of sample"""
        return len(self.target)

    def __getitem__(self, idx):
        """Generate one sample of data"""
        return [self.target[idx], self.features[idx, :]]

    def estimate_qvalues(self, metric: str = "XenithScore",
                         desc: bool = True) \
        -> "Tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)":
        """
        Estimate q-values at the PSM, cross-link, and peptide levels.

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
            The metric by which to rank PSMs. This can either be the
            xenith prediction ("xenith_score") or any feature.

        level : str
            The level at which to estimate q-values. Can be one of
            'psm', 'peptide', or 'cross-link'.

        desc : bool
            Does a higher value of metric indicate a better PSM?

        Returns
        -------
        tuple(pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
            A DataFrame with the q-values at the PSM, peptide, and
            cross-link level, respectively.
        """
        if metric == "XenithScore":
            res_df = self.prediction.XenithScore
        else:
            metric = metric.lower()
            res_df = self._feat_df[metric]

        res_df = pd.concat([self.metadata, res_df], axis=1)

        # Generate keys for grouping
        pep_site1 = res_df.peptidea + "_" + res_df.peptidelinksitea.astype(str)
        pep_site2 = res_df.peptideb + "_" + res_df.peptidelinksiteb.astype(str)
        prot_site1 = (res_df.proteina + "_"
                      + res_df.proteinlinksitea.astype(str))
        prot_site2 = (res_df.proteinb + "_"
                      + res_df.proteinlinksiteb.astype(str))

        res_df["residue_key"] = ["--".join(sorted(x))
                                 for x in zip(prot_site1, prot_site2)]

        res_df["peptide_key"] = ["--".join(sorted(x))
                                 for x in zip(pep_site1, pep_site2)]

        # randomize the df, so that ties are broken randomly
        res_df = res_df.sample(frac=1).reset_index(drop=True)

        psm_cols = ["fileidx", "scannr"]

        # PSM FDR -------------------------------------------------------------
        psm_idx = res_df.groupby(psm_cols)[metric].idxmax()
        psms = res_df.loc[psm_idx, :]

        # Peptide FDR ---------------------------------------------------------
        pep_idx = psms.groupby("peptide_key")[metric].idxmax()
        peps = psms.loc[pep_idx, :]

        # Cross-Link FDR ------------------------------------------------------
        link_idx = peps.groupby("residue_key")[metric].idxmax()
        links = peps.loc[link_idx, :]
        links = links.loc[~links.residue_key.str.contains(";")]

        # Estimate q-values ----------------------------------------------------
        out_list = []
        for dat in (psms, peps, links):
            dat["q-values"] = xenith.fdr.qvalues(dat.numtarget.values,
                                                 dat[metric].values,
                                                 desc=desc)
            dat.sort_values(metric, ascending=(not desc), inplace=True)
            dat.reset_index(drop=True, inplace=True)
            out_list.append(_format_output(dat, metric))

        return out_list

# Functions -------------------------------------------------------------------
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

        logging.info("Assigning fileidx %s to %s.", idx, psm_file)
        psms["fileidx"] = idx

        if not idx:
            feat_set = set(psms.columns)

        if not set(meta_cols) <= set(psms.columns):
            raise RuntimeError(f"{psm_file} does not contain the required "
                               "columns for xenith. See README for details.")

        if not feat_set == set(psms.columns):
            raise RuntimeError(f"Features in {psm_file} do not match the"
                               "features from the previous files.")

        psm_list.append(psms)

    psms = pd.concat(psm_list)
    psms = psms.reset_index(drop=True)
    return psms
