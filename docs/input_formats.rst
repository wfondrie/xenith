Input Formats
=============

PSMs: xenith Tab-Delimited Format
---------------------------------
**xenith** accepts cross-linked PSMs in a tab-delimited format with the following
required metadata columns. Each field is case-insensitive and the order they
appear in the file does not matter:

**PsmId**
   (*string*) A unique identifier for a PSM. This can be anything you want.

**NumTarget**
   (*integer*) The number of target sequences in a cross-linked PSM. This should
   be 0, 1, or 2.

**ScanNr**
   (*integer*) The scan number of the mass spectrum.

**PeptideA**
   (*string*) The peptide sequence for one side of the cross-link. This is used
   to aggregate PSMs to peptides, so it typically includes modifications
   indicated however you prefer.

**PeptideB**
   (*string*) The same as PeptideA, but for the other peptide in the cross-link.

**PeptideLinkSiteA**
   (*integer*) The linked residue on PeptideA.

**PeptideLinkSiteB**
   (*integer*) The linked residue on PeptideB.

**ProteinLinkSiteA**
   (*integer*) The linked residue on PeptideA in the context of the protein
   sequence. If mapped to multiple proteins, all sites should be listed and
   separated with a semi-colon (``;``).

**ProteinLinkSiteB**
   (*integer*) Same as ProteinLinkSiteA, but for PeptideB.

**ProteinA**
   (*string*) The protein(s) which could generate PeptideA. If it could have
   originated from multiple proteins, they should all be listed and separated
   with a semi-colon (``;``).

**ProteinB**
   (*string*) Same as ProteinA, but for PeptideB.

All other columns in the file are assumed to be features unless designated as
additional metadata columns. **xenith** comes with a conversion
tool for Kojak, making it easy to obtain a xenith tab-delimited
file for Kojak_ search results.

.. _Kojak: http://www.kojak-ms.org


Percolator Model Weights
------------------------
If Percolator_ was run on a training dataset, the learned weights can be used as
a pretrained model in **xenith** to re-score a new data set. To accomplish this,
you'll need to tell Percolator_ to save the learned weights to a file.

In the stand-alone version of Percolator_, this is specified using the ``-w`` or
``--weights`` argument:

.. code-block:: console

   $ percolator --weights weights.txt dataset.pin

In `Crux Percolator <http://crux.ms>`_, this is specified using the
``--output-weights`` argument:

.. code-block:: console

   $ crux percolator --output-weights T dataset.pin

.. note::
   In addition to converting Kojak_ search results to **xenith** tab-delimited
   format, **xenith** can convert Kojak_ search results to Percolator
   INput (PIN) format for Percolator. This allows Percolator and **xenith** to use
   the same features.

.. _Percolator: http://percolator.ms
