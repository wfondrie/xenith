"""
Estimate the PSM, peptide, and cross-link level false-discovery rates.
"""
import numpy as np
import pandas as pd

def qvalues(num_targets, metric, desc=True):
    """
    Estimate q-values using the Walzthoeni et al method.

    Parameters
    ----------
    num_targets : numpy.ndarray
        The number of target sequences in the cross-linked pair.

    metric : numpy.ndarray
        The metric to used to rank elements.

    desc : bool
        Is a higher metric better?

    Returns
    -------
    numpy.ndarray
        A 1D array of q-values in the same order as `num_targets` and
        `metric`.
    """
    if metric.shape[0] != num_targets.shape[0]:
        raise ValueError("'metric' and 'num_targets' must be the same length.")

    if desc:
        srt_idx = np.argsort(-metric)
    else:
        srt_idx = np.argsort(metric)

    metric = metric[srt_idx]
    num_targets = num_targets[srt_idx]
    num_total = np.ones(len(num_targets)).cumsum()
    target = (num_targets == 2).astype(int).cumsum()
    one_decoy = (num_targets == 1).astype(int).cumsum()
    two_decoy = (num_targets == 0).astype(int).cumsum()

    fdr = (one_decoy - two_decoy) / target
    fdr[fdr < 0] = 0

    # FDR -> q-values ---------------------------------------------------------
    unique_metric, indices = np.unique(metric, return_counts=True)

    # Flip arrays to loop from worst to best score
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    min_q = 1
    qvals = np.ones(fdr.shape[0])
    group_fdr = np.ones(fdr.shape[0])
    prev_idx = 0
    for idx in range(unique_metric.shape[0]):
        next_idx = prev_idx + indices[idx]
        group = slice(prev_idx, next_idx)
        prev_idx = next_idx

        fdr_group = fdr[group]
        n_group = num_total[group]
        n_max = n_group.max()
        curr_fdr = fdr_group[n_group == n_max]
        if curr_fdr[0] < min_q:
            min_q = curr_fdr[0]

        group_fdr[group] = curr_fdr
        qvals[group] = min_q

    # Restore original order
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals
