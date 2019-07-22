.. xenith documentation master file, created by
   sphinx-quickstart on Fri Jul 12 16:33:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

**xenith** is a python package that uses pretrained models to re-score database
search results for cross-linked peptides from cross-linking mass spectrometry
(XL-MS) experiments. 

Philosophy
----------
In traditional shotgun proteomics, database search post-processing tools such as
Percolator_ and PeptideProphet_ have proven invaluable. By using machine learning
methods to re-score the raw search engine results, these tools dramatically
improve the sensitivity of peptide detection and calibrate the database search
score functions. However, XL-MS experiments present a unique set of challenges.
These include the often limited number of true cross-linked peptide-spectrum matches
(PSMs) and nuances for valid false discovery rate (FDR) estimation.

In light of these challenges, we created **xenith**. The design of **xenith**
follows a traditional machine learning paradigm: Fit a model on a training
dataset [#]_, then use the pretrained model to re-score new datasets of interest.
[#]_
At its most basic, **xenith** provides pretrained models to re-score results
from the Kojak_ search engine. However, **xenith** is also flexible; given an
independent training dataset, weights learned by Percolator can be used to
create a model in **xenith**, or one can be fit within **xenith** itself.
Finally, **xenith** provides the ability to estimate PSM and cross-link level
FDR using the method proposed by Walzthoeni *et al*. [#]_ 

.. _Percolator: http://percolator.ms
.. _PeptideProphet: http://peptideprophet.sourceforge.net/
.. _Kojak: http://www.kojak-ms.org 

An Example
----------
Once **xenith** is installed, it is easy to re-score a new dataset using a
pretrained model. **xenith** offers two operating modes:

+ **From the command line** - Input a dataset, chose a model, then output the
  new scores and q-values for the PSMs and cross-links.

.. code-block:: console
   :caption: Command Line

   $ xenith predict dataset.txt

+ **From Python** - All the functionality of the command line, with additional
  flexibility. Training new models can only be done using **xenith** as a Python
  package.

.. code-block:: python
   :caption: Python

   import xenith
   import pandas as pd
   
   dataset = xenith.load_psms(psm_files="dataset.txt")
   model = xenith.load_model(model="kojak_mlp")

   # Calculate new scores and add them to the dataset.
   new_scores = model.predict(xenith_dataset=dataset)
   dataset.add_metric(values=new_scores, name="score")

   # 'psms' and 'xlinks' are pandas.DataFrames.
   psms, xlinks = dataset.estimate_qvalues(metric="score")


The User Guide
--------------
.. toctree::
   :caption: Documentation
   :maxdepth: 1

   installation.rst
   input_formats.rst
   python_api.rst
   cmd_api.rst

.. toctree::
   :caption: Vignettes
   :maxdepth: 1

   kojak.rst
   training.rst
   


.. rubric:: Footnotes

.. [#] For **xenith**, a dataset is a collection of PSMs from a search engine.
.. [#] The original version of PeptideProphet_ worked in a similar manner.
.. [#] Walzthoeni T, Claassen M, Leitner A, Herzog F, Bohn S, Förster F, Beck M,
       Aebersold R. False discovery rate estimation for cross-linked peptides
       identified by mass spectrometry. Nat Methods. 2012 9(9):901-3. doi:
       10.1038/nmeth.2103. 
