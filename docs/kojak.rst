Re-Score Kojak Results
===============================

**xenith** includes several pretrained models that can be used to re-score the
intraprotein and interprotein cross-linked PSMs from a Kojak_ database search
(versions 2.0 or 1.6.1). If you want to follow along with this tutorial, you'll
need to first perform a database search with Kojak_, specifying that you also
want output for Percolator_. If successful, you should have files ending in
``.kojak.txt``, ``.perc.intra.txt``, and ``.perc.inter.txt``, which you'll need
for **xenith**. In this tutorial, we'll be using ``example.kojak.txt``, etc. as
a generic example.

.. note::
   **xenith** can only be used if the search was performed against a
   concatenated target-decoy sequence database.

Re-scoring Kojak search results requires two steps, which can be performed
either from the command line or within a Python session:

1. Convert Kojak PSM results to **xenith** tab-delimited format.
2. Apply an included pretrained model to re-score the PSMs.


1. Convert Kojak results
------------------------

The Kojak search results must
first be converted to the **xenith** tab-delimited format. In a nutshell, this
conversion combines the interprotein and intraprotein PSMs into a single file,
adding an "intraprotein" feature to the dataset. Additionally, several of the
features in the Kojak_ output for Percolator_ are modified to be better utilized
by **xenith** models. Specifically:

* If E-values are present, they are converted to the -log10(E-Value).
* Charge state is one-hot encoded.
* The "dScore" and "NormRank" are dropped due to their potential to be biased,
  particularly when using non-linear models.
* The "LenShort", "LenLong", and "LenSum" features are linear combinations of
  each other, so "LenShort" and "LenLong" are replaced by "LenRat". This is
  ratio of "LenShort" to "LenLong".
* The "intraprotein" feature is added. Due to the edge case of having a protein
  (and its decoy) in the database, "intraprotein" is only 1 if there are more
  than two proteins detected in the search results.

A new **xenith** tab-delimited file containing the collection PSMs is created.
In this case, it is ``example.xenith.txt``. 

.. code-block:: bash
   :caption: Command Line

   xenith kojak --output_file example.xenith.txt example.kojak.txt example.perc.inter.txt example.perc.intra.txt

.. code-block:: python
   :caption: Python

   import xenith

   xenith.convert.kojak(kojak_txt="example.kojak.txt",
                        perc_inter="example.perc.inter.txt",
                        perc_intra="example.perc.intra.txt",
                        out_file="example.xenith.txt")

2. Re-Score PSMs
-----------------
.. code-block:: bash
   :caption: Command Line

   xenith predict dataset.txt

.. code-block:: python
   :caption: Python

   import xenith
   import pandas as pd

   # Load dataset and model
   dataset = xenith.load_psms("example.xenith.txt")
   model = xenith.load_model("kojak_mlp")

   # Calculate new scores and add them to dataset
   new_scores = model.predict(dataset)
   dataset.add_metric(new_scores)

   # Estimate q-values
   psms, xlinks = dataset.estimate_qvalues()

   # Save PSM and cross-link results
   psms.to_csv("example.psms.txt", sep="\t", index=False)
   xlinks.to_csv("example.xlinks.txt", sep="\t", index=False)


.. _Kojak: http://kojak-ms.org 
.. _Percolator: http://percolator.ms
