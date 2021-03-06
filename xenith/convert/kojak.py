"""
Converter for the Kojak XL-MS search engine output formats.
"""
from itertools import chain
import numpy as np
import pandas as pd

def kojak(kojak_txt: str, perc_inter: str, perc_intra: str, out_file: str,
          version: str = "2.0-dev", max_charge: int = 8,
          decoy_prefix: str = "decoy_", to_pin: bool = False) -> None:
    """
    Convert Kojak search results to xenith tab-delimited format.

    For conversion, the Kojak searches must have been configured to
    output files for Percolator.

    Parameters
    ----------
    kojak_txt : str
        The path to the main kojak result file (.kojak.txt).

    perc_inter : str
        The path to the interprotein Percolator input file from Kojak
        (.perc.inter.txt).

    perc_intra : str
        The path to the intraprotein Percolator input file from Kojak
        (.perc.intra.txt)

    out_file : str
        The path to write the xenith tab-delimited output file.

    version : str
        The Kojak version that results are from.

    max_charge : int
        The maximum charge to consider. This should match the training
        set.

    decoy_prefix : str
        The prefix used to indicate decoy sequences.

    to_pin : bool
        If true, convert results to a Percolator input file instead of
        a xenith tab-delimited file.
    """
    kojak_df, key_cols = _read_kojak(kojak_txt, decoy_prefix)
    inter = _read_percolator(perc_inter)
    inter.SpecId = inter.SpecId + "-inter"

    intra = _read_percolator(perc_intra)
    #intra["intraprotein"] = 1
    intra.SpecId = intra.SpecId + "-intra"

    perc_df = pd.concat([inter, intra])
    merged = pd.merge(perc_df, kojak_df, how="inner", on=key_cols,
                      validate="one_to_one")

    merged["intraprotein"] = _is_intraprotein(merged.ProteinA, merged.ProteinB,
                                              decoy_prefix)
    # In the case where there are only two unique proteins, set the
    # intraprotein feature to 0.
    num_proteins = _count_proteins(merged)
    #if num_proteins <= 2:
    #    merged["intraprotein"] = 0

    # Drop unwanted columns
    xenith_target = ["NumTarget"]
    xenith_tail = ["ProteinLinkSiteA", "ProteinLinkSiteB", "PeptideLinkSiteA",
                   "PeptideLinkSiteB", "ProteinA", "ProteinB", "PeptideA",
                   "PeptideB"]

    perc_target = ["Label"]
    perc_tail = ["Peptide", "Proteins"]

    if to_pin:
        target = perc_target
        tail = perc_tail
        drop_cols = xenith_target + xenith_tail
    else:
        target = xenith_target
        tail = xenith_tail
        drop_cols = perc_target + perc_tail

    # Drop columns specific to Kojak version
    #if version == "2.0-dev":
    #    drop_cols = drop_cols + ["dScore", "NormRank"]

    merged = merged.drop(columns=drop_cols)

    # Reformat E-Values
    e_vals = [col for col in merged.columns if "eVal" in col]
    for val in e_vals:
        merged[val] = -np.log10(merged[val] + np.finfo(np.float64).tiny)

    # Remove linear dependent information
    merged["LenRat"] = (merged.LenShort / merged.LenLong)

    # One-hot encode charge
    for i in range(1, max_charge + 1):
        new_col = merged.Charge == i
        merged[f"Charge_{str(i)}"] = new_col.astype(int)

    merged = merged.drop(columns=["Charge", "LenShort", "LenLong"])

    # Reorder columns for output
    head = ["SpecId"] + target + ["scannr"]
    cols = merged.columns.tolist()
    middle = [col for col in cols if col not in head + tail]
    merged = merged.loc[:, head + middle + tail]
    merged = merged.rename(columns={"SpecId": "PsmId"})

    if not to_pin:
        merged.to_csv(out_file, sep="\t", index=False)
    else:
        _write_pin(merged, out_file)

    return out_file


# Utility Functions -----------------------------------------------------------
def _count_proteins(psm_df):
    """
    Count the number of proteins in the dataset.

    If the number of proteins is 2, intraprotein should be constant.
    """
    all_prot = psm_df.ProteinA + ";" + psm_df.ProteinB

    prot_set = [p.split(";") for p in all_prot.tolist()]
    prot_set = set(chain.from_iterable(prot_set))

    return len(prot_set)


def _write_pin(pin_df, pin_file):
    """
    Write a dataframe to pin format.

    This is only necessary, because pandas *always* must either quote or escape
    the string containing a delimiter.
    """
    with open(pin_file, "w") as pin_out:
        pin_out.write("\t".join(pin_df.columns.tolist()) + "\n")
        for _, row in pin_df.iterrows():
            row = row.values.astype(str)
            pin_out.write("\t".join(row.tolist()) + "\n")


def _read_kojak(kojak_file, decoy_prefix):
    """
    Read a kojak results file and generate key columns

    Parameters
    ----------
    kojak_file : str
        The kojak result file to read.

    decoy_prefix : str
        Decoy prefix string.

    Returns
    -------
    tuple(pandas.DataFrame, list)
        A dataframe containing the parsed PSMs and a list naming the
        key columns to join with the Percolator data.
    """
    dat = pd.read_csv(kojak_file, sep="\t", skiprows=1)
    dat = dat.loc[dat["Protein #2"] != "-"]
    key_cols = ["scannr", "Peptide", "Label"]
    pep1 = dat["Peptide #1"].str.replace(r"\[.+?\]", "")
    pep2 = dat["Peptide #2"].str.replace(r"\[.+?\]", "")
    link1 = dat["Linked AA #1"]
    link2 = dat["Linked AA #2"]
    decoy1 = _all_decoy(dat["Protein #1"], decoy_prefix)
    decoy2 = _all_decoy(dat["Protein #2"], decoy_prefix)
    dat["Protein #1"] = _parse_proteins(dat["Protein #1"])
    dat["Protein #2"] = _parse_proteins(dat["Protein #2"])

    dat["scannr"] = dat["Scan Number"]
    dat["Peptide"] = ("-." + pep1 + "(" + link1 + ")--"
                      + pep2 + "(" + link2 + ").-")

    dat["Label"] = (((decoy1.values - 1) * (decoy2.values - 1))*2 - 1)

    # rename some columns for the final file
    dat["NumTarget"] = (decoy1.values + decoy2.values - 2) * -1
    dat = dat.rename(columns={"Protein #1 Site": "ProteinLinkSiteA",
                              "Protein #2 Site": "ProteinLinkSiteB",
                              "Linked AA #1": "PeptideLinkSiteA",
                              "Linked AA #2": "PeptideLinkSiteB",
                              "Protein #1": "ProteinA",
                              "Protein #2": "ProteinB",
                              "Peptide #1": "PeptideA",
                              "Peptide #2": "PeptideB"})

    final_cols = ["NumTarget", "ProteinA", "ProteinB", "PeptideA", "PeptideB",
                  "ProteinLinkSiteA", "ProteinLinkSiteB",
                  "PeptideLinkSiteA", "PeptideLinkSiteB"]

    dat = dat.loc[:, key_cols + final_cols]
    return (dat, key_cols)


def _read_percolator(percolator_file):
    """Parse a PIN formatted file. Return a dataframe"""
    with open(percolator_file, "r") as pin:
        header = pin.readline()
        splits = header.count("\t")
        header = header.replace("\n", "")
        header = header.split("\t")
        rows = [line.replace("\n", "").split("\t", splits) for line in pin]

    data = pd.DataFrame(columns=header, data=rows)
    return data.apply(pd.to_numeric, errors="ignore")


def _parse_proteins(protein_col):
    """Remove description from protein id."""
    protein_col = protein_col.str.split(";")
    prot = [";".join([p.strip().split(" ", 1)[0] for p in r]) for r in protein_col]

    return prot

def _is_intraprotein(protein_col_a, protein_col_b, decoy_prefix):
    """Determine if the cross-link is between the same protein or it's decoy"""
    protein_col_a = protein_col_a.str.replace(decoy_prefix, "").str.split(";")
    protein_col_b = protein_col_b.str.replace(decoy_prefix, "").str.split(";")
    return [int(set(a) == set(b)) for a, b in zip(protein_col_a, protein_col_b)]

def _all_decoy(protein_col, decoy_prefix):
    """Returns 1 if all proteins are decoys, 0 otherwise."""
    ret = []
    protein_col = protein_col.str.split(";")
    for row in protein_col:
        decoy = all([p.startswith(decoy_prefix) for p in row])
        ret.append(decoy)

    return pd.Series(ret).astype(int)
