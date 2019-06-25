
<img src="xenith_logo.svg" width=200>  
---  
[![Build Status](https://travis-ci.org/wfondrie/xenith.svg?branch=master)](https://travis-ci.org/wfondrie/xenith)
[![codecov](https://codecov.io/gh/wfondrie/xenith/branch/master/graph/badge.svg)](https://codecov.io/gh/wfondrie/xenith)
[![Built with Spacemacs](https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg)](http://spacemacs.org)  

Enhanced cross-linked peptide detection using pretrained models.  

**xenith** is a python package that uses pretrained models to re-score search
results from database search engines for cross-linked peptides. It also provides
the ability to train custom models or import the learned weights from 
[Percolator](percolator.ms).

## Installation
**xenith** is not yet available through PyPI or conda, but it can be installed
directly from GitHub and is compatible with Python 3.6+.

### Dependencies
**xenith** has the following dependencies:  

+ NumPy  
+ Pandas  
+ PyTorch  

Missing dependencies, with the exception of PyTorch, will be automatically
installed when you install **xenith** with `pip` or `conda`. The PyTorch 
installation depends on your specific GPU version, if you have one.
Please refer to the PyTorch installation instructions for more information.

### Installing xenith with pip
**xenith** can be easily installed using `pip`:  
```bash
pip install git+https://github.com/wfondrie/xenith.git
```
### Installing xenith with conda
**xenith** cannot yet be installed with conda.

## Usage  
### xenith tab-delimited format  
**xenith** accepts cross-linked PSMs in a tab-delimited format with the 
following required metadata columns. Each field is case-insensitive and 
the order they appear in the file does not matter:  

+ `PsmId` - (string) A unique identifier for a PSM. This can be anything
you want.  
+ `NumTarget` - (integer) The number of target sequences in a cross-linked
PSM. This should be 0, 1, or 2.  
+ `ScanNr` - (integer) The scan number of the mass spectrum.  
+ `PeptideA` - (string) The peptide sequence for one side of the cross-link.
This is used to aggregate PSMs to peptides, so it typically includes 
modifications indicated however you prefer.  
+ `PeptideB` - (string) The same as `PeptideA`, but for the other peptide in the
cross-link.  
+ `PeptideLinkSiteA` - (integer) The linked residue on `PeptideA`.  
+ `PeptideLinkSiteB` - (integer) The linked residue on `PeptideB`.  
+ `ProteinLinkSiteA` - (integer) The linked residue on `PeptideA` in the 
context of the protein sequence. If mapped to multiple proteins, all sites 
should be listed and separated with a semi-colon (`;`).  
+ `ProteinLinkSiteB` - (integer) Same as `ProteinLinkSiteA`, but for `PeptideB`.  
+ `ProteinA` - (string) The protein(s) that `PeptideA` maps too. If it could
have originated from multiple proteins, they should all be listed and separated
with a semi-colon (`;`).  
+ `ProteinB` - (string) Same as `ProteinA`, but for `PeptideB`.  

All other columns in the file are interpreted as features for the selected model.
It is critical that the feature columns match the features from the selected 
model.

For convenience, **xenith** includes a converter for search results from Kojak 
2.0.  

### An example in Python  
Convert search results from Kojak 2.0 to xenith format:
```Python
import xenith
import pandas as pd

# Convert Kojak results to xenith format:
tsv = xenith.convert_kojak(kojak="example.kojak.txt",
                           perc_inter="example.perc.inter.txt",
                           perc_intra="example.perc.intra.txt",
                           out_file="example.xenith.tsv")
```
We can train a new model. For this you will need two datasets in addition to
the dataset you are interested in. The `training_files` will be used to train 
the model and the `validation_files` will be used to assess the model during
training:  

```Python
# Train a new multilayer perceptron model: 
# "train.tsv"- A collection of PSMs to train the model.
# "val.tsv" - A collection of PSMs to monitor model training.

model = xenith.new_model(num_features=24)
model.fit(training_files="train.tsv", validation_files="val.tsv")

# Save the model:
model.save("xenith_model.pt")
```

Finally, we can use the pretrained model to asses a new dataset:  
```Python
# Load the model
model = xenith.load("xenith_model.pt")

# Rescore using the model
predictions = model.predict(psm_files="example.xenith.tsv")

# Estimate qvalues
# 'psms', 'peptides' and 'crosslinks' are each a pandas.DataFrame
psms, peptides, crosslinks = predictions.estimate_qvalues("xenith_score")

# Save PSM-level results
psms.to_csv("psms.tsv", sep="\t", index=False)
```

### TODO  
- [ ] Finish command line API.
- [ ] Implement formal tests and Travis-CI.
- [ ] Documentation (readthedocs).
- [ ] PSI output formats.
- [ ] ProXL output.

