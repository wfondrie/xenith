
<img src="images/xenith_logo.svg" width=200>  

---  

[![Build Status](https://travis-ci.org/wfondrie/xenith.svg?branch=master)](https://travis-ci.org/wfondrie/xenith)
[![codecov](https://codecov.io/gh/wfondrie/xenith/branch/master/graph/badge.svg)](https://codecov.io/gh/wfondrie/xenith)
[![Documentation
Status](https://readthedocs.org/projects/xenith/badge/?version=latest)](https://xenith.readthedocs.io/en/latest/?badge=latest)
![Python 3.6](https://img.shields.io/badge/python-3.6-brightgreen.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-brightgreen.svg)
[![Built with Spacemacs](https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg)](http://spacemacs.org)  

Enhanced cross-linked peptide detection using pretrained models.  

**xenith** is a python package that uses pretrained models to re-score database
search results for cross-linked peptides from cross-linking mass spectrometry
(XL-MS) experiments.

## Philosophy  
In traditional shotgun proteomics, database search post-processing tools such as
Percolator and PeptideProphet have proven invaluable. By using machine learning
methods to re-score the raw search engine results, these tools dramatically
improve the sensitivity of peptide detection and calibrate the database search
score functions. However, XL-MS experiments present a unique set of challenges.
These include the often limited number of true cross-linked peptide-spectrum
matches (PSMs) and nuances for valid false discovery rate (FDR) estimation.  

In light of these challenges, we created xenith. The design of xenith follows a
traditional machine learning paradigm: Fit a model on a training dataset 1, then
use the pretrained model to re-score new datasets of interest. 2 At its most
basic, xenith provides pretrained models to re-score results from the Kojak
search engine. However, xenith is also flexible; given an independent training
dataset, weights learned by Percolator can be used to create a model in xenith,
or one can be fit within xenith itself. Finally, xenith provides the ability to
estimate PSM and cross-link level FDR using the method proposed by [*Walzthoeni et
al*](https://www.nature.com/articles/nmeth.2103).

## An Example  

Once xenith is installed, it is easy to re-score a new dataset using a
pretrained model. xenith offers two operating modes:

* **From the command line** - Input a dataset, chose a model, then output the new
  scores and q-values for the PSMs and cross-links. 

```console

$ xenith predict dataset.txt

```

* **From Python** - All the functionality of the command line, with additional
  flexibility. Training new models can only be done using **xenith** as a Python
  package.

```python
import xenith
import pandas as pd

dataset = xenith.load_psms(psm_files="dataset.txt")
model = xenith.load_model(model="kojak_mlp")

# Calculate new scores and add them to the dataset.
new_scores = model.predict(xenith_dataset=dataset)
dataset.add_metric(values=new_scores, name="score")

# 'psms' and 'xlinks' are pandas.DataFrames.
psms, xlinks = dataset.estimate_qvalues(metric="score")
```

For more information, check out the documentation site: https://xenith.readthedocs.io
